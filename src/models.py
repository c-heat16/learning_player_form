__author__ = 'Connor Heaton'

import json
import torch
import random

import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss

from losses import SupConLoss
from vocabularies import GeneralGamestateDeltaVocab


class RawDataProjectionModule(nn.Module):
    """
    Simple class for creating a multi-layer module that projects an input vector to a specified smaller dimension.
    """
    def __init__(self, in_dim, out_dim, n_layers, dropout_p=0.15, activation_fn='relu', do_layernorm=False,
                 suppress_final_activation=False, hidden_dim=None, elementwise_affine=False,
                 do_batchnorm=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.activation_fn = activation_fn
        self.do_layernorm = do_layernorm
        self.do_batchnorm = do_batchnorm
        self.suppress_final_activation = suppress_final_activation
        self.hidden_dim = hidden_dim if hidden_dim is not None else in_dim

        self.dropout = nn.Dropout(self.dropout_p)
        if self.activation_fn == 'relu':
            self.activation = nn.ReLU()

        for i in range(self.n_layers):
            if self.do_layernorm and i > 0:
                setattr(self, 'proj_layernorm_{}'.format(i), nn.LayerNorm([self.hidden_dim],
                                                                          elementwise_affine=elementwise_affine))
            elif self.do_batchnorm and i > 0:
                setattr(self, 'proj_bn_{}'.format(i), nn.BatchNorm1d(self.hidden_dim))

            if i == 0:
                setattr(self, 'proj_layer_{}'.format(i), nn.Linear(self.in_dim, self.hidden_dim))
            elif i < self.n_layers - 1:
                setattr(self, 'proj_layer_{}'.format(i), nn.Linear(self.hidden_dim, self.hidden_dim))
            else:
                setattr(self, 'proj_layer_{}'.format(i), nn.Linear(self.hidden_dim, out_dim))

            layer = getattr(self, 'proj_layer_{}'.format(i))
            layer.bias.data.zero_()
            layer.weight.data.normal_(mean=0.0, std=0.01)

    def forward(self, x):
        for i in range(self.n_layers):
            # if i != 0:
            x = self.dropout(x)

            # if self.do_layernorm and 0 < i < (self.n_layers - 1):
            if self.do_layernorm and i > 0:
                norm_layer = getattr(self, 'proj_layernorm_{}'.format(i))
                x = norm_layer(x)
            elif self.do_batchnorm and i > 0:
                norm_layer = getattr(self, 'proj_bn_{}'.format(i))
                x = norm_layer(x)

            # print('i: {} x: {}'.format(i, x.shape))
            proj_layer = getattr(self, 'proj_layer_{}'.format(i))
            # print('proj_layer: {}'.format(proj_layer.weight.shape))
            x = proj_layer(x)

            if self.suppress_final_activation and i == self.n_layers - 1:
                x = x
            else:
                x = self.activation(x)

        return x


class PlayerFormModel(nn.Module):
    def __init__(self, args):
        super(PlayerFormModel, self).__init__()
        self.args = args
        self.verbosity = getattr(self.args, 'verbosity', 0)
        self.loss_objective = getattr(self.args, 'loss_objective', 'supcon')

        self.n_pitch_types = 19
        self.gamestate_embd_dim = getattr(self.args, 'gamestate_embd_dim', 128)
        self.max_seq_len = getattr(self.args, 'max_seq_len', 130)
        self.general_dropout_prob = getattr(self.args, 'general_dropout_prob', 0.15)
        self.token_mask_pct = getattr(self.args, 'token_mask_pct', 0.0)
        self.mask_override_prob = getattr(self.args, 'mask_override_prob', 0.15)
        self.pad_mask_pct = getattr(self.args, 'pad_mask_pct', 0.0)
        # self.gs_token_predict_pct = getattr(self.args, 'gs_token_predict_pct', 0.15)
        self.n_transformer_layers = getattr(self.args, 'n_layers', 6)
        self.n_attn = getattr(self.args, 'n_attn', 8)
        self.n_raw_data_proj_layers = getattr(self.args, 'n_raw_data_proj_layers', 2)
        self.player_type = getattr(self.args, 'player_type', 'batter')
        self.parse_plate_pos_to_id = getattr(self.args, 'parse_plate_pos_to_id', False)

        self.form_ab_window_size = getattr(self.args, 'form_ab_window_size', 25)
        self.n_ab_pitch_no_embds = getattr(self.args, 'n_ab_pitch_no_embds', 25)

        self.player_pos_embd_dim = getattr(self.args, 'player_pos_embd_dim', 10)
        self.raw_pitcher_data_dim = getattr(self.args, 'raw_pitcher_data_dim', 216)
        self.raw_batter_data_dim = getattr(self.args, 'raw_batter_data_dim', 216)
        self.raw_matchup_data_dim = getattr(self.args, 'raw_matchup_data_dim', 243)
        self.complete_embd_dim = getattr(self.args, 'complete_embd_dim', 256)
        self.pitch_type_embd_dim = getattr(self.args, 'pitch_type_embd_dim', 15)
        self.plate_pos_embd_dim = getattr(self.args, 'plate_pos_embd_dim', 5)
        self.n_plate_x_ids = getattr(self.args, 'n_plate_x_ids', 8)
        self.n_plate_z_ids = getattr(self.args, 'n_plate_z_ids', 9)
        self.proj_dim = getattr(self.args, 'proj_dim', 96)
        self.n_stadium_ids = getattr(self.args, 'n_stadium_ids', 40)
        self.stadium_embd_dim = getattr(self.args, 'stadium_embd_dim', 10)
        self.handedness_embd_dim = getattr(self.args, 'handedness_embd_dim', 5)
        self.use_handedness = getattr(self.args, 'use_handedness', True)
        self.both_player_positions = getattr(self.args, 'both_player_positions', False)
        self.both_player_handedness = getattr(self.args, 'both_player_handedness', True)
        # self.first_principles = getattr(self.args, 'first_principles', False)
        # self.use_statcast_data = getattr(self.args, 'use_statcast_data', True)
        # self.use_player_data = getattr(self.args, 'use_player_data', True)

        print('********** PlayerFormModel **********')
        self.supplemental_input_dim = self.raw_pitcher_data_dim + self.raw_batter_data_dim \
                                      + self.raw_matchup_data_dim
        self.supplemental_output_dim = self.complete_embd_dim - self.gamestate_embd_dim - self.pitch_type_embd_dim - \
                                       self.player_pos_embd_dim - 17 - self.stadium_embd_dim

        if self.parse_plate_pos_to_id:
            print('$$$ Learning embeddings for positions over plate $$$')
            self.supplemental_output_dim -= 2 * self.plate_pos_embd_dim
        else:
            self.supplemental_output_dim -= 2

        if self.both_player_positions:
            print('$$$ Using both player position embeddings $$$')
            self.supplemental_output_dim -= self.player_pos_embd_dim

        if self.use_handedness:
            if self.both_player_handedness:
                print('$$$ Using both player handedness $$$')
                self.supplemental_output_dim -= (2 * self.handedness_embd_dim)
            else:
                print('$$$ Using one player handedness $$$')
                self.supplemental_output_dim -= self.handedness_embd_dim

        print('Supplemental - in dim: {} out dim: {}'.format(self.supplemental_input_dim,
                                                             self.supplemental_output_dim))
        print('# raw data proj layers: {}'.format(self.n_raw_data_proj_layers))
        print('Gamestate embed dim: {}'.format(self.gamestate_embd_dim))
        print('Complete embed dim: {}'.format(self.complete_embd_dim))
        print('# layers: {} # attn: {}'.format(self.n_transformer_layers, self.n_attn))
        print('*' * len('********** PlayerFormModel **********'))

        # Gamestate vocab parms
        self.gamestate_vocab_bos_inning_no = getattr(self.args, 'gamestate_vocab_bos_inning_no', True)
        self.gamestate_vocab_bos_score_diff = getattr(self.args, 'gamestate_vocab_bos_score_diff', True)
        self.gamestate_vocab_use_balls_strikes = getattr(self.args, 'gamestate_vocab_use_balls_strikes', True)
        self.gamestate_vocab_use_base_occupancy = getattr(self.args, 'gamestate_vocab_use_base_occupancy', True)
        self.gamestate_vocab_use_inning_no = getattr(self.args, 'gamestate_vocab_use_inning_no', False)
        self.gamestate_vocab_use_inning_topbot = getattr(self.args, 'gamestate_vocab_use_inning_topbot', False)
        self.gamestate_vocab_use_score_diff = getattr(self.args, 'gamestate_vocab_use_score_diff', True)
        self.gamestate_vocab_use_outs = getattr(self.args, 'gamestate_vocab_use_outs', True)
        self.gamestate_n_innings = getattr(self.args, 'gamestate_n_innings', 10)
        self.gamestate_max_score_diff = getattr(self.args, 'gamestate_max_score_diff', 6)
        self.gamestate_vocab = GeneralGamestateDeltaVocab(bos_inning_no=self.gamestate_vocab_bos_inning_no,
                                                          max_inning_no=self.gamestate_n_innings,
                                                          bos_score_diff=self.gamestate_vocab_bos_score_diff,
                                                          bos_max_score_diff=self.gamestate_max_score_diff,
                                                          balls_delta=self.gamestate_vocab_use_balls_strikes,
                                                          strikes_delta=self.gamestate_vocab_use_balls_strikes,
                                                          outs_delta=self.gamestate_vocab_use_outs,
                                                          score_delta=self.gamestate_vocab_use_score_diff,
                                                          base_occ_delta=self.gamestate_vocab_use_base_occupancy
                                                          )
        self.n_gamestate_tokens = len(self.gamestate_vocab)
        self.n_gamestate_bos_tokens = len(self.gamestate_vocab.bos_vocab)
        self.mask_id = self.gamestate_vocab.mask_id

        # Info about player positions
        self.player_pos_source = getattr(self.args, 'player_pos_source', 'mlb')
        if self.player_pos_source == 'mlb':
            self.player_pos_key = 'mlb_pos'
        elif self.player_pos_source == 'cbs':
            self.player_pos_key = 'cbs_pos'
        elif self.player_pos_source == 'espn':
            self.player_pos_key = 'espn_pos'
        pos_map_fp = getattr(self.args, '{}_id_map_fp'.format(self.player_pos_key),
                             '../config/{}_mapping.json'.format(self.player_pos_key))
        self.player_pos_id_map = json.load(open(pos_map_fp))
        self.n_pos_embds = len(self.player_pos_id_map)

        # create & init embeddings
        self.gamestate_embds = nn.Embedding(self.n_gamestate_tokens, self.gamestate_embd_dim, max_norm=None)
        self.cls_embd = nn.Embedding(1, self.complete_embd_dim, max_norm=None)
        self.positional_embeddings = nn.Embedding(self.max_seq_len, self.complete_embd_dim, max_norm=None)
        self.ab_pitch_no_embds = nn.Embedding(self.n_ab_pitch_no_embds, self.complete_embd_dim, max_norm=None)
        self.ab_idx_embds = nn.Embedding(self.form_ab_window_size, self.complete_embd_dim, max_norm=None)
        self.stadium_embds = nn.Embedding(self.n_stadium_ids, self.stadium_embd_dim, max_norm=None)
        nn.init.uniform_(self.gamestate_embds.weight, -0.01, 0.01)
        nn.init.uniform_(self.cls_embd.weight, -0.01, 0.01)
        nn.init.uniform_(self.ab_pitch_no_embds.weight, -0.01, 0.01)
        nn.init.uniform_(self.ab_idx_embds.weight, -0.01, 0.01)
        nn.init.uniform_(self.positional_embeddings.weight, -0.01, 0.01)
        nn.init.uniform_(self.stadium_embds.weight, -0.01, 0.01)

        # player-related parms
        self.player_position_embds = nn.Embedding(self.n_pos_embds + 1, self.player_pos_embd_dim, max_norm=None)
        nn.init.uniform_(self.player_position_embds.weight, -0.01, 0.01)

        self.supplemental_projection_module = RawDataProjectionModule(
            in_dim=self.supplemental_input_dim,
            out_dim=self.supplemental_output_dim,
            n_layers=self.n_raw_data_proj_layers,
            dropout_p=self.general_dropout_prob,
            do_layernorm=True
        )
        self.supplemental_layernorm = nn.LayerNorm([self.supplemental_output_dim], elementwise_affine=False)

        # statcast-related parms
        self.pitch_type_embds = nn.Embedding(self.n_pitch_types, self.pitch_type_embd_dim, max_norm=None)
        nn.init.uniform_(self.pitch_type_embds.weight, -0.01, 0.01)

        if self.parse_plate_pos_to_id:
            self.plate_x_pos_embds = nn.Embedding(self.n_plate_x_ids, self.plate_pos_embd_dim, max_norm=None)
            self.plate_z_pos_embds = nn.Embedding(self.n_plate_z_ids, self.plate_pos_embd_dim, max_norm=None)

            nn.init.uniform_(self.plate_x_pos_embds.weight, -0.01, 0.01)
            nn.init.uniform_(self.plate_z_pos_embds.weight, -0.01, 0.01)

        if self.use_handedness:
            self.handedness_embds = nn.Embedding(2, self.handedness_embd_dim, max_norm=None)
            nn.init.uniform_(self.handedness_embds.weight, -0.01, 0.01)

        # core modeling parms
        transformer_layernorm = nn.LayerNorm([self.complete_embd_dim], elementwise_affine=False)
        encoder_layer = nn.TransformerEncoderLayer(self.complete_embd_dim,
                                                   self.n_attn,
                                                   4 * self.complete_embd_dim,
                                                   self.general_dropout_prob)
        self.transformer = nn.TransformerEncoder(encoder_layer, self.n_transformer_layers, norm=transformer_layernorm)

        self.dropout = nn.Dropout(self.general_dropout_prob)
        if self.gs_token_predict_pct > 0 or self.token_mask_pct:
            self.gamestate_clf_head = nn.Linear(self.complete_embd_dim, self.n_gamestate_tokens)

        self.n_proj_layers = getattr(self.args, 'n_proj_layers', 2)
        self.proj_head = RawDataProjectionModule(
            in_dim=self.complete_embd_dim,
            out_dim=self.proj_dim,
            n_layers=self.n_proj_layers,
            dropout_p=self.general_dropout_prob,
            do_layernorm=True
        )

    def forward(
            self, gamestate_ids, pitcher_data, batter_data, matchup_data,
            pitcher_pos_ids=None, batter_pos_ids=None, gamestate_labels=None, pitch_type_ids=None,
            plate_x_ids=None, plate_z_ids=None, rv_pitch_data=None, model_src_pad_mask=None, ab_lengths=None,
            do_id_mask=False, record_ab_numbers=None, pitch_numbers=None, stadium_ids=None, pitcher_handedness=None,
            batter_handedness=None
    ):
        """
        Simple wrapper around the self.process_data(...) method. Used to compute loss once data is processed
        :return: gamestate prediction loss, contrastive learning loss
        """
        cls_to_return = 'proj'
        processed_data = self.process_data(gamestate_ids=gamestate_ids, pitcher_data=pitcher_data,
                                           batter_data=batter_data, matchup_data=matchup_data,
                                           pitcher_pos_ids=pitcher_pos_ids, batter_pos_ids=batter_pos_ids,
                                           gamestate_labels=gamestate_labels, pitch_type_ids=pitch_type_ids,
                                           plate_x_ids=plate_x_ids, plate_z_ids=plate_z_ids,
                                           rv_pitch_data=rv_pitch_data,
                                           model_src_pad_mask=model_src_pad_mask, ab_lengths=ab_lengths,
                                           do_id_mask=do_id_mask, record_ab_numbers=record_ab_numbers,
                                           pitch_numbers=pitch_numbers,
                                           stadium_ids=stadium_ids, pitcher_handedness=pitcher_handedness,
                                           batter_handedness=batter_handedness,
                                           cls_to_return=cls_to_return,
                                           )

        cls_projections, (gamestate_logits, gamestate_labels) = processed_data

        gamestate_loss = None
        con_loss_fn = SupConLoss()
        con_loss = con_loss_fn(cls_projections)

        if gamestate_labels is not None and self.token_mask_pct > 0:
            xent_loss_fn = CrossEntropyLoss()
            gamestate_loss = xent_loss_fn(gamestate_logits, gamestate_labels)

        outputs = (gamestate_loss, con_loss)
        return outputs

    def process_data(
            self, gamestate_ids, pitcher_data, batter_data, matchup_data,
            pitcher_pos_ids=None, batter_pos_ids=None, gamestate_labels=None, pitch_type_ids=None,
            plate_x_ids=None, plate_z_ids=None, rv_pitch_data=None, model_src_pad_mask=None, ab_lengths=None,
            do_id_mask=False, record_ab_numbers=None, pitch_numbers=None, stadium_ids=None, cls_to_return='proj',
            normalize_cls_proj=True, pitcher_handedness=None, batter_handedness=None,
    ):
        """
        Process a batch of data. Handles masking and making predictions if warranted.

        :param gamestate_ids: set of IDs describing in-game events in the sequence
        :param pitcher_data: data describing the pitcher taking part in the at-bats
        :param batter_data: data describing batter taking part in the at-bats
        :param matchup_data: data describing historical matchup between pitcher and batter taking part in the at-bats
        :param pitcher_pos_ids: position ID of pitcher taking part in the at-bats
        :param batter_pos_ids: position ID of batter taking part in the at-bats
        :param gamestate_labels: labels of gamestate sequence to be predicted. Should match gamestate_ids initially
        :param pitch_type_ids: IDs for types of pitches thrown
        :param plate_x_ids: x location of pitches over the plate
        :param plate_z_ids: z location of pitches over the place
        :param rv_pitch_data: real-valued data describing the thrown pitches and batted ball if applicable
        :param model_src_pad_mask: indicates where 'real' gamestate delta tokens are present in the input
        :param ab_lengths: length of the at-bats in the input sequences
        :param do_id_mask: boolean indicating if the input sequences should be matched
        :param record_ab_numbers: at-bat numbers within sequence. used to construct positional embeddings
        :param pitch_numbers: pitch numbers within each at-bat. used to construct positional embeddings
        :param stadium_ids: ID's denoting the stadium in which the at-bats take place
        :param cls_to_return: the type of [CLS] token to return. 'proj' returns the form vector, 'embd' returns the
        processed [CLS] token as is.
        :param normalize_cls_proj: whether or not the form vector should be normalized
        :param pitcher_handedness: handedness of the pitchers taking part in the at-bats
        :param batter_handedness: handedness of the batters taking part in the at-bats
        :return: the specified version of the processed [CLS] token, and (gamestate predictions, gamestate labels)
        """
        if len(gamestate_ids.shape) == 3:
            batch_size = gamestate_ids.shape[0]
            n_views = gamestate_ids.shape[1]
            seq_len = gamestate_ids.shape[2]
        else:
            batch_size = gamestate_ids.shape[0]
            n_views = 1
            seq_len = gamestate_ids.shape[1]

        src_mask = torch.zeros((seq_len + 1, seq_len + 1)).to(gamestate_ids.device)

        gamestate_ids = gamestate_ids.view(batch_size * n_views, seq_len)
        pitcher_pos_ids = pitcher_pos_ids.view(batch_size * n_views, seq_len)
        batter_pos_ids = batter_pos_ids.view(batch_size * n_views, seq_len)

        pitch_type_ids = pitch_type_ids.view(batch_size * n_views, seq_len)
        plate_x_ids = plate_x_ids.view(batch_size * n_views, seq_len)
        plate_z_ids = plate_z_ids.view(batch_size * n_views, seq_len)

        model_src_pad_mask = model_src_pad_mask.view(batch_size * n_views, seq_len)
        record_ab_numbers = record_ab_numbers.view(batch_size * n_views, seq_len)
        pitch_numbers = pitch_numbers.view(batch_size * n_views, seq_len)
        stadium_ids = stadium_ids.view(batch_size * n_views, seq_len)
        ab_lengths = ab_lengths.view(batch_size * n_views, -1)

        pitcher_data = pitcher_data.view(batch_size * n_views, seq_len, -1)
        batter_data = batter_data.view(batch_size * n_views, seq_len, -1)
        matchup_data = matchup_data.view(batch_size * n_views, seq_len, -1)
        rv_pitch_data = rv_pitch_data.view(batch_size * n_views, seq_len, -1)

        mask_idxs = None
        if do_id_mask and self.token_mask_pct > 0:
            gamestate_ids, mask_idxs = self.mask_input_sequence(gamestate_ids, ab_lengths)

        gamestate_embds = self.gamestate_embds(gamestate_ids)
        stadium_embds = self.stadium_embds(stadium_ids)
        input_seq_components = [gamestate_embds, stadium_embds]

        if pitcher_handedness is not None and batter_handedness is not None and self.use_handedness:
            pitcher_handedness = pitcher_handedness.view(batch_size * n_views, seq_len)
            batter_handedness = batter_handedness.view(batch_size * n_views, seq_len)
            pitcher_handendess_embds = self.handedness_embds(pitcher_handedness)
            batter_handendess_embds = self.handedness_embds(batter_handedness)

            if self.both_player_handedness:
                input_seq_components.extend([pitcher_handendess_embds, batter_handendess_embds])
            elif getattr(self, 'player_type', 'batter') == 'batter':
                input_seq_components.extend([pitcher_handendess_embds])
            else:
                input_seq_components.extend([batter_handendess_embds])

        pitcher_pos_embds = self.player_position_embds(pitcher_pos_ids)
        batter_pos_embds = self.player_position_embds(batter_pos_ids)

        if getattr(self, 'both_player_positions', False):
            input_seq_components.extend([pitcher_pos_embds, batter_pos_embds])
        elif getattr(self, 'player_type', 'batter') == 'batter':
            input_seq_components.extend([pitcher_pos_embds])
        else:
            input_seq_components.extend([batter_pos_embds])

        supplemental_inputs = self.process_supplemental_inputs(pitcher_data, batter_data, matchup_data)
        input_seq_components.extend([supplemental_inputs])

        pitch_type_embds = self.pitch_type_embds(pitch_type_ids)
        if self.parse_plate_pos_to_id:
            plate_x_embds = self.plate_x_pos_embds(plate_x_ids.int())
            plate_z_embds = self.plate_z_pos_embds(plate_z_ids.int())
        else:
            plate_x_embds = plate_x_ids.unsqueeze(-1)
            plate_z_embds = plate_z_ids.unsqueeze(-1)

        input_seq_components.extend([pitch_type_embds, plate_x_embds, plate_z_embds, rv_pitch_data])
        input_seq = torch.cat(input_seq_components, dim=-1)

        # add positional encodings
        record_ab_embds = self.ab_idx_embds(record_ab_numbers)
        pitch_number_embds = self.ab_pitch_no_embds(pitch_numbers)
        input_seq = input_seq + record_ab_embds + pitch_number_embds

        cls_ids = torch.tensor([0 for _ in range(input_seq.shape[0])]).unsqueeze(1).to(input_seq.device)
        cls_embds = self.cls_embd(cls_ids)
        input_seq = torch.cat([input_seq, cls_embds], dim=1)
        model_src_pad_mask_adj = torch.zeros(model_src_pad_mask.shape[0], 1,
                                             dtype=torch.bool).to(model_src_pad_mask.device)
        model_src_pad_mask = torch.cat([model_src_pad_mask, model_src_pad_mask_adj], dim=1)
        input_seq = input_seq.transpose(0, 1)
        output_seq = self.transformer(input_seq, mask=src_mask, src_key_padding_mask=model_src_pad_mask)
        output_seq = output_seq.transpose(0, 1)
        cls_embds = output_seq[:, -1]

        cls_embds = self.dropout(cls_embds)
        if cls_to_return == 'proj':
            cls_projections = self.proj_head(cls_embds)
            if normalize_cls_proj:
                cls_projections = F.normalize(cls_projections, dim=-1)
            cls_projections = cls_projections.view(batch_size, n_views, -1)

        gamestate_logits = None
        output_seq = output_seq[:, :-1]
        if gamestate_labels is not None and mask_idxs is not None:
            gamestate_labels = gamestate_labels.view(-1, seq_len)

            output_seq, gamestate_labels = self.select_gamestate_embeddings_for_prediction(output_seq,
                                                                                           gamestate_labels,
                                                                                           mask_idxs)
            gamestate_logits = self.gamestate_clf_head(output_seq)

        if cls_to_return == 'proj':
            to_return = (cls_projections, (gamestate_logits, gamestate_labels))
        else:
            to_return = (cls_embds, (gamestate_logits, gamestate_labels))

        return to_return

    def mask_input_sequence(self, input_seq, ab_lengths):
        """
        Randomly mask a certain percent of input tokens for the model to predict. Will respect the given ab_lengths
        parm as to mask *real* input tokens, not the padding.
        :param input_seq: input sequence(s) to be masked
        :param ab_lengths: lengths of at-bats in each sequence. Used to identify valid candidate indices for masking
        :return: masked input sequence
        """
        all_mask_idxs = []

        for batch_idx in range(input_seq.shape[0]):
            if sum(ab_lengths[batch_idx]) >= input_seq.shape[-1]:
                print('*' * 60)
                print('sum(ab_lengths[batch_idx]) ({}) >= input_seq.shape[-1] ({})'.format(sum(ab_lengths[batch_idx]),
                                                                                           input_seq.shape[-1]))
                print('*' * 60)
            this_item_n_pitches = min(sum(ab_lengths[batch_idx]), input_seq.shape[-1])
            potential_mask_idxs = [i for i in range(this_item_n_pitches)]
            random.shuffle(potential_mask_idxs)

            mask_idxs = potential_mask_idxs[:int(self.token_mask_pct * len(potential_mask_idxs))]
            all_mask_idxs.append(mask_idxs)
            for mask_idx in mask_idxs:
                if random.random() > self.mask_override_prob:
                    input_seq[batch_idx, mask_idx] = self.mask_id

        return input_seq, all_mask_idxs

    def process_supplemental_inputs(self, pitcher_data, batter_data, matchup_data):
        """
        Process the supplemental inputs for a given batch. Dimensions of each set up supplemental data should match
        those given in the parms

        :param pitcher_data: supplemental data describing the pitcher
        :param batter_data: supplemental data describing the batter
        :param matchup_data: supplemental data describing the matchup
        :return: processed supplemental data
        """
        supplemental_proj_inputs = torch.cat([pitcher_data, batter_data, matchup_data], dim=-1)
        supplemental_proj = self.supplemental_projection_module(supplemental_proj_inputs)
        supplemental_proj = self.supplemental_layernorm(supplemental_proj)

        return supplemental_proj




