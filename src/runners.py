__author__ = 'Connor Heaton'

import os
import math
import time
import torch

import torch.distributed as dist

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import get_constant_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import PlayerFormDataset
from models import PlayerFormModel
from vocabularies import GeneralGamestateDeltaVocab


class PlayerFormRunner(object):
    """
    A 'Runner' class which handles the training of a model.
    """
    def __init__(self, gpu, mode, args):
        """
        Creates a runner object in accordance with the given arguments and mode on the requested device
        :param gpu: the index of the GPU to place the model on
        :param mode: the mode of operation. train/dev/test/apply
        :param args: arguments to be used during running
        """
        self.rank = gpu
        self.mode = mode
        self.args = args
        self.ab_data_cache = {}

        print('Initializing PlayerFormRunner_v2 on device {}...'.format(gpu))
        if self.args.on_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.rank))
            torch.cuda.set_device(self.device)

        print('\ttorch.cuda.device_count(): {}'.format(torch.cuda.device_count()))
        torch.manual_seed(self.args.seed)
        dist.init_process_group('nccl',
                                world_size=len(self.args.gpus),
                                rank=self.rank)

        # Model parms & settings
        self.version = getattr(self.args, 'version', 'v1')
        self.lr = getattr(self.args, 'lr', 1e-5)
        self.l2 = getattr(self.args, 'l2', 0.0001)
        # self.supplemental_l2 = getattr(self.args, 'l2', 0.0001)
        self.max_seq_len = getattr(self.args, 'max_seq_len', 12)
        self.log_every = getattr(self.args, 'log_every', 10)
        self.save_model_every = getattr(self.args, 'save_model_every', -1)
        self.save_preds_every = getattr(self.args, 'save_preds_every', 5)
        self.pred_dir = os.path.join(self.args.out, 'preds')
        self.log_dir = os.path.join(self.args.out, 'logs')
        self.n_warmup_iters = getattr(self.args, 'n_warmup_iters', -1)
        self.gs_token_predict_pct = getattr(self.args, 'gs_token_predict_pct', 0.15)

        self.complete_embd_dim = getattr(self.args, 'complete_embd_dim', 512)
        self.raw_mgsm_weight = getattr(self.args, 'mgsm_weight', 1.0)
        self.raw_con_weight = getattr(self.args, 'con_weight', 1.0)
        if self.raw_con_weight <= 0:
            self.mgsm_weight = 1.0
            self.con_weight = 0.0
        elif self.raw_mgsm_weight <= 0:
            self.mgsm_weight = 0.0
            self.con_weight = 1.0
        else:
            self.mgsm_weight = self.raw_mgsm_weight / (self.raw_mgsm_weight + self.raw_con_weight)
            self.con_weight = self.raw_con_weight / (self.raw_mgsm_weight + self.raw_con_weight)

        if self.rank == 0:
            print('*' * 50)
            print('MGSM Weight: {}'.format(self.mgsm_weight))
            print('Contrast Weight: {}'.format(self.con_weight))
            print('*' * 50)

        self.use_handedness = getattr(self.args, 'use_handedness', False)

        print('PlayerFormRunner on device {} creating gamestate vocab...'.format(self.rank))
        gamestate_vocab_bos_inning_no = getattr(self.args, 'gamestate_vocab_bos_inning_no', True)
        gamestate_vocab_bos_score_diff = getattr(self.args, 'gamestate_vocab_bos_score_diff', True)
        gamestate_vocab_use_balls_strikes = getattr(self.args, 'gamestate_vocab_use_balls_strikes', True)
        gamestate_vocab_use_base_occupancy = getattr(self.args, 'gamestate_vocab_use_base_occupancy', True)
        gamestate_vocab_use_score_diff = getattr(self.args, 'gamestate_vocab_use_score_diff', True)
        gamestate_vocab_use_outs = getattr(self.args, 'gamestate_vocab_use_outs', True)
        gamestate_n_innings = getattr(self.args, 'gamestate_n_innings', 10)
        gamestate_max_score_diff = getattr(self.args, 'gamestate_max_score_diff', 6)
        self.gamestate_vocab = GeneralGamestateDeltaVocab(bos_inning_no=gamestate_vocab_bos_inning_no,
                                                          max_inning_no=gamestate_n_innings,
                                                          bos_score_diff=gamestate_vocab_bos_score_diff,
                                                          bos_max_score_diff=gamestate_max_score_diff,
                                                          balls_delta=gamestate_vocab_use_balls_strikes,
                                                          strikes_delta=gamestate_vocab_use_balls_strikes,
                                                          outs_delta=gamestate_vocab_use_outs,
                                                          score_delta=gamestate_vocab_use_score_diff,
                                                          base_occ_delta=gamestate_vocab_use_base_occupancy,
                                                          )

        if not os.path.exists(self.pred_dir) and self.rank == 0:
            os.makedirs(self.pred_dir)
        self.pred_fp_tmplt = os.path.join(self.pred_dir, '{}_preds_epoch_{}.csv')
        self.log_fp_tmplt = os.path.join(self.log_dir, '{}_logs_epoch_{}.csv')

        print('PlayerFormRunner_v2 on device {} creating dataset...'.format(self.rank))

        self.dataset = PlayerFormDataset(self.args, self.mode, self.gamestate_vocab, ab_data_cache=self.ab_data_cache)
        if self.args.on_cpu:
            data_sampler = None
        else:
            data_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset,
                                                                           num_replicas=args.world_size,
                                                                           rank=self.rank,
                                                                           shuffle=True,
                                                                           )
        self.data_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=False,
                                      num_workers=self.args.n_data_workers, pin_memory=True, sampler=data_sampler,
                                      drop_last=True, persistent_workers=True)
        self.n_iters = int(math.ceil(len(self.dataset) / (self.args.batch_size * len(self.args.gpus))))

        self.aux_dataset = None
        self.aux_data_loader = None
        self.aux_n_iters = None
        if self.mode == 'train' and (self.args.dev or self.args.dev_every > 0):
            print('PlayerFormRunner_v2 on device {} creating auxiliary dataset...'.format(self.rank))
            self.aux_dataset = PlayerFormDataset(self.args, 'dev', self.gamestate_vocab,
                                                 ab_data_cache=self.ab_data_cache)

            if self.args.on_cpu:
                aux_data_sampler = None
            else:
                aux_data_sampler = torch.utils.data.distributed.DistributedSampler(self.aux_dataset,
                                                                                   num_replicas=args.world_size,
                                                                                   rank=self.rank,
                                                                                   shuffle=False)

            self.aux_data_loader = DataLoader(self.aux_dataset, batch_size=self.args.batch_size, shuffle=False,
                                              num_workers=int(2 * self.args.n_data_workers), pin_memory=True,
                                              sampler=aux_data_sampler, drop_last=False, persistent_workers=True)
            self.aux_n_iters = int(math.ceil(len(self.aux_dataset) / (self.args.batch_size * len(self.args.gpus))))

        print('PlayerFormRunner on device {} creating model...'.format(self.rank))
        self.model = PlayerFormModel(self.args)
        self.model = self.model.to(self.device)

        self.start_epoch = 0
        if self.mode != 'train':
            # ckpt_file = self.args.out_dir
            if self.args.train:
                ckpt_epoch_offset = 1
                ckpt_file = os.path.join(self.args.model_save_dir,
                                         self.args.ckpt_file_tmplt.format(self.args.epochs - ckpt_epoch_offset))
                while not os.path.exists(ckpt_file) and self.args.epochs - ckpt_epoch_offset >= 0:
                    ckpt_epoch_offset += 1
                    ckpt_file = os.path.join(self.args.model_save_dir,
                                             self.args.ckpt_file_tmplt.format(self.args.epochs - ckpt_epoch_offset))
            else:
                ckpt_file = self.args.ckpt_file
        else:
            ckpt_file = self.args.ckpt_file

        if self.rank == 0:
            print('*** ckpt_file: {} ***'.format(ckpt_file))
        if ckpt_file is not None:
            if self.rank == 0:
                print('Loading model from ckpt...')
                print('\tckpt_file: {}'.format(ckpt_file))
            map_location = {'cuda:{}'.format(0): 'cuda:{}'.format(gpu_id) for gpu_id in args.gpus}
            state_dict = torch.load(ckpt_file, map_location=map_location)
            self.model.load_state_dict(state_dict, strict=False)

            if ckpt_file == self.args.ckpt_file:
                self.start_epoch = int(ckpt_file.split('_')[-1][:-4]) + 1

        if not self.args.on_cpu:
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)

        self.summary_writer = None
        if self.rank == 0 and self.mode == 'train':
            self.summary_writer = SummaryWriter(log_dir=self.args.tb_dir)

        self.scheduler = None
        self.n_epochs = 1
        if self.mode == 'train':
            self.n_epochs = self.args.epochs

            no_decay = ['layernorm', 'norm']
            param_optimizer = list(self.model.named_parameters())
            no_decay_parms = []
            reg_parms = []
            for n, p in param_optimizer:
                if any(nd in n for nd in no_decay):
                    no_decay_parms.append(p)
                else:
                    reg_parms.append(p)

            optimizer_grouped_parameters = [
                {'params': reg_parms, 'weight_decay': self.l2},
                {'params': no_decay_parms, 'weight_decay': 0.0},
            ]
            if self.rank == 0:
                print('n parms: {}'.format(len(param_optimizer)))
                print('len(optimizer_grouped_parameters[0]): {}'.format(len(optimizer_grouped_parameters[0]['params'])))
                print('len(optimizer_grouped_parameters[1]): {}'.format(len(optimizer_grouped_parameters[1]['params'])))
            self.optimizer = optim.Adam(optimizer_grouped_parameters, lr=self.lr, betas=(0.9, 0.95))
            if self.n_warmup_iters > 0:
                self.scheduler = get_constant_schedule_with_warmup(self.optimizer,
                                                                   num_warmup_steps=self.n_warmup_iters)

        self.run()

    def run(self):
        for epoch in range(self.start_epoch, self.n_epochs):
            if self.rank == 0:
                print('Performing epoch {} of {}'.format(epoch, self.n_epochs))

            if self.mode == 'train':
                self.model.train()
            else:
                self.model.eval()

            if self.mode == 'train':
                self.run_one_epoch(epoch, self.mode)
            else:
                with torch.no_grad():
                    self.run_one_epoch(epoch, self.mode)

            if self.mode == 'train':
                if epoch % self.save_model_every == 0 and self.save_model_every > 0:
                    if self.rank == 0:
                        print('Saving model...')
                        if not self.args.on_cpu:
                            torch.save(self.model.module.state_dict(),
                                       os.path.join(self.args.model_save_dir, self.args.ckpt_file_tmplt.format(epoch)))
                        else:
                            torch.save(self.model.state_dict(),
                                       os.path.join(self.args.model_save_dir, self.args.ckpt_file_tmplt.format(epoch)))
                    dist.barrier()

                if self.args.dev_every > 0 and epoch % self.args.dev_every == 0:
                    self.model.eval()
                    with torch.no_grad():
                        self.run_one_epoch(epoch, 'train-dev')

    def run_one_epoch(self, epoch, mode):
        if mode == self.mode:
            dataset = self.data_loader
            n_iters = self.n_iters
        else:
            dataset = self.aux_data_loader
            n_iters = self.aux_n_iters

        iter_since_grad_accum = 1
        last_batch_end_time = None
        write_lines = []
        for batch_idx, batch_data in enumerate(dataset):
            global_item_idx = (epoch * n_iters) + batch_idx
            batch_start_time = time.time()

            sample_file_start_idx = batch_data['sample_file_start_idx']
            sample_file_end_idx = batch_data['sample_file_end_idx']
            n_player_files = batch_data['n_player_files']

            inning_ids = batch_data['inning_ids'].to(self.device, non_blocking=True)
            ab_number_ids = batch_data['ab_number_ids'].to(self.device, non_blocking=True)
            state_delta_ids = batch_data['state_delta_ids'].to(self.device, non_blocking=True)
            ab_lengths = batch_data['ab_lengths'].to(self.device, non_blocking=True)
            pitch_types = batch_data['pitch_types'].to(self.device, non_blocking=True)
            plate_x = batch_data['plate_x'].to(self.device, non_blocking=True)
            plate_z = batch_data['plate_z'].to(self.device, non_blocking=True)
            pitcher_inputs = batch_data['pitcher_inputs'].to(self.device, non_blocking=True)
            batter_inputs = batch_data['batter_inputs'].to(self.device, non_blocking=True)
            matchup_inputs = batch_data['matchup_inputs'].to(self.device, non_blocking=True)
            pitcher_id = batch_data['pitcher_id'].to(self.device, non_blocking=True)
            batter_id = batch_data['batter_id'].to(self.device, non_blocking=True)
            pitcher_pos_ids = batch_data['pitcher_pos_ids'].to(self.device, non_blocking=True)
            batter_pos_ids = batch_data['batter_pos_ids'].to(self.device, non_blocking=True)
            rv_pitch_data = batch_data['rv_pitch_data'].to(self.device, non_blocking=True)
            pitch_numbers = batch_data['pitch_numbers'].to(self.device, non_blocking=True)
            record_ab_numbers = batch_data['record_ab_numbers'].to(self.device, non_blocking=True)
            my_src_pad_mask = batch_data['my_src_pad_mask'].to(self.device, non_blocking=True)
            model_src_pad_mask = batch_data['model_src_pad_mask'].to(self.device, non_blocking=True)
            stadium_ids = batch_data['stadium_ids'].to(self.device, non_blocking=True)
            pitcher_handedness_ids = batch_data['pitcher_handedness_ids'].to(self.device, non_blocking=True)
            batter_handedness_ids = batch_data['batter_handedness_ids'].to(self.device, non_blocking=True)
            gamestate_labels = state_delta_ids[:]

            if batch_idx == 0 and epoch == 0 and self.rank == 0:
                print('state_delta_ids: {}'.format(state_delta_ids.shape))
                print('ab_number_ids: {}'.format(ab_number_ids.shape))
                print('inning_ids: {}'.format(inning_ids.shape))
                print('pitcher_inputs: {}'.format(pitcher_inputs.shape))
                print('pitcher_inputs: {}'.format(pitcher_inputs.shape))
                print('batter_inputs: {}'.format(batter_inputs.shape))
                print('batter_inputs: {}'.format(batter_inputs.shape))
                print('matchup_inputs: {}'.format(matchup_inputs.shape))
                print('matchup_inputs: {}'.format(matchup_inputs.shape))
                print('pitcher_pos_ids: {}'.format(pitcher_pos_ids.shape))
                print('batter_pos_ids: {}'.format(batter_pos_ids.shape))
                print('gamestate_labels: {}'.format(gamestate_labels.shape))
                print('pitch_types: {}'.format(pitch_types.shape))
                print('plate_x: {}'.format(plate_x.shape))
                print('plate_z: {}'.format(plate_z.shape))
                print('rv_pitch_data: {}'.format(rv_pitch_data.shape))
                print('my_src_pad_mask: {}'.format(my_src_pad_mask.shape))
                print('model_src_pad_mask: {}'.format(model_src_pad_mask.shape))
                print('pitcher_id: {}'.format(pitcher_id.shape))
                print('batter_id: {}'.format(batter_id.shape))
                print('pitch_numbers: {}'.format(pitch_numbers.shape))
                print('record_ab_numbers: {}'.format(record_ab_numbers.shape))
                print('ab_lengths: {}'.format(ab_lengths.shape))
                print('stadium_ids: {}'.format(stadium_ids.shape))

                if self.use_handedness:
                    print('pitcher_handedness_ids: {}'.format(pitcher_handedness_ids.shape))
                    print('batter_handedness_ids: {}'.format(batter_handedness_ids.shape))

                print('plate_z - min: {} max: {}'.format(plate_z.min(), plate_z.max()))
                print('plate_x - min: {} max: {}'.format(plate_x.min(), plate_x.max()))

            model_outputs = self.model(
                state_delta_ids, pitcher_inputs, batter_inputs, matchup_inputs, pitcher_pos_ids, batter_pos_ids,
                gamestate_labels, pitch_types, plate_x, plate_z, rv_pitch_data, model_src_pad_mask, ab_lengths,
                do_id_mask=True if mode == 'train' else False, record_ab_numbers=record_ab_numbers,
                pitch_numbers=pitch_numbers, stadium_ids=stadium_ids, pitcher_handedness=pitcher_handedness_ids,
                batter_handedness=batter_handedness_ids,
            )

            gamestate_loss = model_outputs[0]
            con_loss = model_outputs[1]

            if gamestate_loss is not None:
                total_loss = (self.mgsm_weight * gamestate_loss) + (self.con_weight * con_loss)
            else:
                total_loss = con_loss

            if mode == 'train':
                total_loss.backward()

            ab_lengths = ab_lengths.detach().cpu()
            batter_id = batter_id.detach().cpu()
            pitcher_id = pitcher_id.detach().cpu()

            if epoch % self.log_every == 0:
                if batch_idx == 0:
                    write_lines.append('player_id,n_total_abs,ab_start_idx,ab_end_idx,n_pitches_view1,n_pitches_view2')
                for record_idx in range(sample_file_start_idx.shape[0]):
                    if self.args.player_type == 'batter':
                        player_id = batter_id[record_idx, 0, 0].item()
                    else:
                        player_id = pitcher_id[record_idx, 0, 0].item()

                    player_n_files = n_player_files[record_idx, 0, 0].item()
                    ab_start_idx = sample_file_start_idx[record_idx, 0, 0].item()
                    ab_end_idx = sample_file_end_idx[record_idx, 0, 0].item()
                    n_pitches_view1 = ab_lengths[record_idx, 0, :].sum().item()
                    n_pitches_view2 = ab_lengths[record_idx, 1, :].sum().item()
                    write_items = [player_id, player_n_files, ab_start_idx, ab_end_idx, n_pitches_view1,
                                   n_pitches_view2]
                    write_line = ','.join([str(v) for v in write_items])
                    write_lines.append(write_line)

            if global_item_idx % self.args.grad_summary_every == 0 and self.summary_writer is not None \
                    and mode == 'train' and self.args.grad_summary and global_item_idx != 0:

                for name, p in self.model.named_parameters():
                    if p.grad is not None and p.grad.data is not None:
                        self.summary_writer.add_histogram('grad/{}'.format(name), p.grad.data,
                                                          (epoch * n_iters) + batch_idx)
                        self.summary_writer.add_histogram('weight/{}'.format(name), p.data,
                                                          (epoch * n_iters) + batch_idx)

            if global_item_idx % self.args.print_every == 0 and self.rank == 0:
                batch_elapsed_time = time.time() - batch_start_time
                if last_batch_end_time is not None:
                    time_btw_batches = batch_start_time - last_batch_end_time
                else:
                    time_btw_batches = 0.0

                print_str = '{0}- Epoch: {1}/{2} Iter: {3}/{4} Total: {5:.4f} Gamestate: {6:.4f} ' \
                            'Contrast: {7:.4f} Time: {8:.2f}s ({9:.2f}s)'
                print_str = print_str.format(mode, epoch, self.n_epochs, batch_idx, n_iters, total_loss,
                                             gamestate_loss if gamestate_loss is not None else -1.0,
                                             con_loss if con_loss is not None else -1.0,
                                             batch_elapsed_time, time_btw_batches)
                print(print_str)
                last_batch_end_time = time.time()

            if (global_item_idx % self.args.summary_every == 0 and self.summary_writer is not None) \
                    or (mode == 'train-dev' and self.summary_writer is not None
                        and global_item_idx % int(self.args.summary_every / 3) == 0):

                if total_loss is not None:
                    self.summary_writer.add_scalar('total_loss/{}'.format(mode), total_loss,
                                                   (epoch * n_iters) + batch_idx)
                if gamestate_loss is not None:
                    self.summary_writer.add_scalar('gamestate_loss/{}'.format(mode), gamestate_loss,
                                                   (epoch * n_iters) + batch_idx)
                if con_loss is not None:
                    self.summary_writer.add_scalar('con_loss/{}'.format(mode), con_loss,
                                                   (epoch * n_iters) + batch_idx)

            if iter_since_grad_accum == self.args.n_grad_accum and mode == 'train':
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                iter_since_grad_accum = 1
            else:
                iter_since_grad_accum += 1

        if iter_since_grad_accum > 1 and mode == 'train':
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

        if len(write_lines) > 1 and self.rank == 0:
            print('Writing logs to file...')
            log_fp = self.log_fp_tmplt.format(mode, epoch)
            with open(log_fp, 'w+') as f:
                f.write('\n'.join(write_lines))






