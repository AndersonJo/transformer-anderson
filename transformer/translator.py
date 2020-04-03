from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Transformer
from transformer.mask import create_nopeak_mask, create_padding_mask


class Translator(nn.Module):
    def __init__(self, model: Transformer, beam_size: int, device: torch.device, max_seq_len: int,
                 src_pad_idx: int, trg_pad_idx: int, trg_sos_idx: int, trg_eos_idx: int):
        super(Translator, self).__init__()

        self.model = model
        self.model.eval()

        self.beam_size = beam_size
        self.device = device
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.trg_eos_idx = trg_eos_idx

        # init_trg_seq: [["<sos>"]] 로 시작하는 matrix 이며, output 의 초기값으로 사용됨
        # beam_output: beam search 를 하기 위해서 decoder에서 나온 output 값들을 저장한다
        init_trg_seq = torch.LongTensor([[self.trg_sos_idx]]).to(self.device)
        beam_output = torch.full((self.beam_size, self.max_seq_len), self.trg_pad_idx, dtype=torch.long)
        beam_output[:, 0] = self.trg_sos_idx
        beam_output = beam_output.to(self.device)

        self.register_buffer('init_trg_seq', init_trg_seq)
        self.register_buffer('beam_output', beam_output)

    def _create_init_sequence(self, src_seq) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_mask = create_padding_mask(src_seq, pad_idx=self.src_pad_idx)
        src_mask = src_mask.to(self.device)

        # Encoder
        # 먼저 source sentence tensor 를 encoder 에 집어넣고 encoder output 을 냅니다
        enc_output = self.model.encoder(src_seq, src_mask)  # (1, seq_len, embed_size)

        # Decoder
        trg_mask = create_nopeak_mask(self.init_trg_seq)  # (1, 1, 1)
        dec_output = self.model.decoder(self.init_trg_seq, trg_mask, enc_output, src_mask)  # (1, 1, embed_size)
        dec_output = self.model.out_linear(dec_output)  # (1, 1, 9473)
        dec_output = F.softmax(dec_output, dim=-1)  # (1, 1, 9473) everything is zero except one element

        k_probs, k_indices = dec_output[:, -1, :].topk(self.beam_size)
        scores = torch.log1p(k_probs).view(self.beam_size)

        # Generate beam sequences
        beam_output = self.beam_output.clone().detach()  # (beam_size, max_seq_len)
        beam_output[:, 1] = k_indices[0]

        # Reshape encoder output
        enc_output = enc_output.repeat(self.beam_size, 1, 1)  # (beam_size, seq_len, embed_dim)
        return enc_output, beam_output, scores

    def beam_search(self, src_seq):
        """
        :param src_seq:
        :return:
        """

        # Create initial source padding mask
        dec_output = self._create_init_sequence(src_seq)

    def translate(self, src_seq: torch.LongTensor):
        assert src_seq.size(0) == 1  # Batch Size should be 1

        with torch.no_grad():
            self.beam_search(src_seq)
            import ipdb
            ipdb.set_trace()
