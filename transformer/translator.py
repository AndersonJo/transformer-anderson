from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import Vocab

from transformer import Transformer
from transformer.debug import to_sentence
from transformer.mask import create_nopeak_mask, create_padding_mask


class Translator(nn.Module):
    def __init__(self, model: Transformer, beam_size: int, device: torch.device, max_seq_len: int,
                 src_pad_idx: int, trg_pad_idx: int, trg_sos_idx: int, trg_eos_idx: int,
                 src_vocab: Vocab, trg_vocab: Vocab):
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

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        # init_trg_seq: [["<sos>"]] 로 시작하는 matrix 이며, output 의 초기값으로 사용됨
        # beam_output: beam search 를 하기 위해서 decoder에서 나온 output 값들을 저장한다
        init_trg_seq = torch.LongTensor([[self.trg_sos_idx]]).to(self.device)
        seq_arange = torch.arange(1, self.max_seq_len + 1, dtype=torch.long).to(self.device)
        beam_output = torch.full((self.beam_size, self.max_seq_len), self.trg_pad_idx, dtype=torch.long)
        beam_output[:, 0] = self.trg_sos_idx
        beam_output = beam_output.to(self.device)

        self.register_buffer('init_trg_seq', init_trg_seq)
        self.register_buffer('beam_output', beam_output)
        self.register_buffer('seq_arange', seq_arange)

    def _create_init_sequence(self,
                              src_seq: torch.Tensor,
                              src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param src_seq: (1, seq_size)
        :param src_mask: (1, 1, seq_size)
        :return:
            - enc_output: (beam_size, seq_len, embed_dim)
            - beam_output: (beam_size, max_seq_len)
            - scores: (beam_size,)
        """
        # Encoder
        # 먼저 source sentence tensor 를 encoder 에 집어넣고 encoder output 을 냅니다
        enc_output = self.model.encoder(src_seq, src_mask)  # (1, seq_len, embed_size)

        # Decoder
        dec_output = self._decoder_softmax(self.init_trg_seq, enc_output, src_mask)
        k_probs, k_indices = dec_output[:, -1, :].topk(self.beam_size)
        scores = torch.log1p(k_probs).view(self.beam_size)

        # Generate beam sequences
        beam_output = self.beam_output.clone().detach()  # (beam_size, max_seq_len)
        beam_output[:, 1] = k_indices[0]

        # Reshape encoder output
        enc_output = enc_output.repeat(self.beam_size, 1, 1)  # (beam_size, seq_len, embed_dim)
        return enc_output, beam_output, scores

    def _decoder_softmax(self,
                         trg_seq: torch.Tensor,
                         enc_output: torch.Tensor,
                         src_mask: torch.Tensor) -> torch.Tensor:
        trg_mask = create_nopeak_mask(trg_seq)  # (1, 1, 1)
        dec_output = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)  # (1, 1, embed_size)
        dec_output = self.model.out_linear(dec_output)  # (1, 1, 9473)
        dec_output = F.softmax(dec_output, dim=-1)  # (1, 1, 9473) everything is zero except one element
        return dec_output

    def _calculate_scores(self,
                          step: int,
                          beam_output: torch.Tensor,
                          dec_output: torch.Tensor,
                          scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param step: start from 2 to max_sequence or until reaching to the <eos>
        :param beam_output: (beam_size, max_seq_len)
        :param dec_output: (beam_size, 2~step, word_size) ex.(5, 2, 9473) -> next -> (5, 3, 9473)
        :param scores: (beam_size, beam_size) ex. (5, 5)
        :return:
        """
        # k_probs: (beam_size, beam_size == topk) -> (5, 5) -> 가장 마지막으로 예측한 단어 top k 의 단어 확률
        # k_indices: (beam_size, beam_size) -> (5, 5) -> 가장 마지막으로 예측한 단어 top k 의 index
        k_probs, k_indices = dec_output[:, -1, :].topk(self.beam_size)

        # Calculate scores added by previous scores
        # 즉 단어 하나하나 생성하며서 문장을 만들어 나가는데..
        # 누적 점수를 만들어 나가면서 나중에 가장 점수가 높은 문장을 찾아내겠다는 의미
        # scores: (beam_size, beam_size)  ex. (5, 5)
        scores = torch.log1p(k_probs) + scores

        # (5, 5) 에서 만들어진 전체 단어중에서 best k 단어를 찾아낸다
        # beam_scores: (5*5,)
        # beam_indices: (5*5,) index는 0에서 25사이의 값이다. (즉 단어의 index가 아니다)
        beam_scores, beam_indices = scores.view(-1).topk(self.beam_size)
        _row_idx = beam_indices // self.beam_size
        _col_idx = beam_indices % self.beam_size
        best_indices = k_indices[_row_idx, _col_idx]  # (beam_size,) k_indices안에 단어의 index가 들어있다

        # best_indices 와 row index와 동일하게 맞쳐준후, best indices 값을 추가한다
        beam_output[:, :step] = beam_output[_row_idx, :step]
        beam_output[:, step] = best_indices

        return beam_scores, beam_output

    def beam_search(self, src_seq):
        """
        beam_output 설명
            기본적으로 beam_output 은 다음과 같이 생겼다
            tensor([[   2, 1615,    1,    1,    1],
                    [   2,  538,    1,    1,    1],
                    [   2,    2,    1,    1,    1],
                    [   2,    1,    1,    1,    1],
                    [   2,    0,    1,    1,    1]]
            여기서 2="<sos>", 1="<pad>" 이며, 5개의 beam_size에서 forloop 을 돌면서, 그 다음 단어를 예측하며,
            "<eos>" 가 나올때까지 padding 부분을 단어 index 로 채워 나가면서 계속 진행한다

        :param src_seq:
        :return:
        """
        # Create initial source padding mask
        src_mask = create_padding_mask(src_seq, pad_idx=self.src_pad_idx)
        src_mask = src_mask.to(self.device)
        enc_output, beam_output, scores = self._create_init_sequence(src_seq, src_mask)
        ans_row_idx = 0
        for step in range(2, self.max_seq_len):
            dec_output = self._decoder_softmax(beam_output[:, :step], enc_output, src_mask)
            scores, beam_output = self._calculate_scores(step, beam_output, dec_output, scores)

            # Find complete setences and end this loop
            eos_loc = beam_output == self.trg_eos_idx  # (beam_size, max_seq_size) ex. (5, 100)

            # (beam_size, max_seq_size) 에서 대부분 max_seq_len 값을 갖고, trg_eos 만 실제 index값을 갖는다
            eos_indices, _ = self.seq_arange.masked_fill(~eos_loc, self.max_seq_len).min(1)

            n_complete_sentences = (eos_loc.sum(1) > 0).sum().item()

            # DEBUG
            # print(to_sentence(src_seq[0], self.src_vocab))
            # for i in range(5):
            #     print(to_sentence(beam_output[i], self.src_vocab)[:150])

            if n_complete_sentences == self.beam_size:
                ans_row_idx = scores.max(0)[1].item()
                break

        return beam_output[ans_row_idx][:eos_indices[ans_row_idx]].tolist()

    def translate(self, src_seq: torch.LongTensor):
        assert src_seq.size(0) == 1  # Batch Size should be 1

        with torch.no_grad():
            pred_seq = self.beam_search(src_seq)
        return pred_seq
