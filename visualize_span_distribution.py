import matplotlib.pyplot as plt
import numpy as np
import json
from natsort import natsorted


def get_span_list(json_files):
    all_spans = []

    for i in json_files:
        with open(i, 'r') as f:
            all_spans += json.load(f)
            f.close()

    return all_spans


def get_freq(all_spans, seq_len, first_n_frame_dynamics):
    freq = [0] * seq_len
    sample_cnt = 0

    for sample in all_spans:
        sample_cnt += 1
        span_set = set()
        for spans in sample:
            for i in spans:
                span_set.add(i)
        span_list = natsorted(list(span_set))
        for idx in span_list:
            freq[idx-first_n_frame_dynamics-1] += 1

    for i in range(len(freq)):
        freq[i] /= sample_cnt

    return freq


first_n_frame_dynamics = 3
max_seq_len = 40
seq_len = max_seq_len - first_n_frame_dynamics
save_file = "span_distribution.png"
all_spans = get_span_list(['experiments/2021-09-09 04-42-45 contain train pip/epoch_1/train/all_spans.json'])


freq = get_freq(all_spans, seq_len, first_n_frame_dynamics)
x = np.linspace(first_n_frame_dynamics+1, max_seq_len, num=seq_len)
y = np.asarray(freq)
plt.plot(x, y)
plt.title("Span Selection Frequency")
plt.xlabel("Generated Frame")
plt.ylabel("Selection Frequency")
plt.yticks(ticks=[0,0.2,0.4,0.6,0.8,1.0])
plt.grid()
plt.show()
plt.savefig(save_file)