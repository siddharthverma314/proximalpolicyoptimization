import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('Creates a plot of the information in a log file')
parser.add_argument('log_dir')
parser.add_argument('--log-plot', action='store_true')
args = parser.parse_args()


# open file
logdir = args.log_dir
if logdir[-1] != '/':
    logdir += '/'


# collect data
data = []
with open(f"{logdir}/log") as f:
    for line in f:
        if line.startswith('INFO') and 'Epoch' in line:
            data.append({'reward': [],})
        elif line.startswith('DEBUG') and 'reward' in line:
            num = float(line.split()[-1])
            data[-1]['reward'].append(num)

# plot data
R = []
for epoch in data:
    r = np.array(epoch['reward'])
    R.append(r)
R = np.row_stack(R)

means = R.mean(axis=1)
stds = R.std(axis=1)
x = np.arange(len(means)) + 1
plt.plot(x, means, 'r-')
plt.fill_between(x, means-stds, means+stds, facecolor="#FF0000", alpha=0.2)
plt.savefig(f"{logdir}/plot.png")
plt.show()

if args.log_plot:
    plt.plot(np.log(x), means, 'r-')
    plt.fill_between(np.log(x), means-stds, means+stds, facecolor="#FF0000", alpha=0.2)
    plt.savefig(f"{logdir}/plot_log.png")
    plt.show()
