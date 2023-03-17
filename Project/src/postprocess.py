import re
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import get_cmap


LOGPATH = "./log.log"


def process_train_log(path):

    loss_by_task = {}

    with open(path, mode="r") as r:

        accuracy_results = False
        metrics = {}
        task_counter = 1
        prev_epoch = 0
        task_loss = []        

        for line in r.readlines():
            if(not line):
                continue
            if(line.startswith("-")):
                accuracy_results = True

            if(not accuracy_results):
                m = re.match(r"\[(\d+)/70\] mean_loss : (.*).*", line)
                if(m):

                    current_epoch = int(m.group(1))
                    current_loss = float(m.group(2))

                    if(current_epoch < prev_epoch):

                        loss_by_task[task_counter] = task_loss
                        task_counter+=1
                        task_loss = []
                        prev_epoch = current_epoch
                    else:

                        task_loss.append(current_loss)
                        prev_epoch = current_epoch
                else:
                    pass
            else:
                
                
                m = re.match(r".*qa(\d+)(.*)", line)
                if(m):
                    task = int(m.group(1))
                    eval_accuracy = m.group(2).split()[-1]
                    metrics[task] = {"eval_accuracy": float(eval_accuracy)}

        loss_by_task[task_counter] = task_loss

        for k in loss_by_task.keys():
            test_idx = tuple(range(0, len(loss_by_task[k]), 2))
            eval_idx = tuple(range(1, len(loss_by_task[k]), 2))

            test_loss = tuple(map(lambda x: loss_by_task[k][x], test_idx))
            eval_loss = tuple(map(lambda x: loss_by_task[k][x], eval_idx))

            loss_by_task[k] = {"train_loss": test_loss, "eval_loss": eval_loss}

            metrics[k]["train_loss"] = loss_by_task[k]["train_loss"]
            metrics[k]["eval_loss"] = loss_by_task[k]["eval_loss"]

        return metrics



def plot_tasks(results, task_indexes, title,
               metrics=["train_loss", "eval_loss",],
               linestyles={"train": "-", "eval": "--"},
               outfile=None, show=True):

    clean_names = []
    for task in task_indexes:
        for metric in metrics:

            metric_name = " ".join(map(lambda x: x.capitalize(), metric.split("_")))
            if(metric_name not in clean_names):
                clean_names.append(metric_name)
            metric_hist = results[task][metric]

            color = "red" if "train" in metric else "blue"
            if(len(task_indexes) > 1):
                color = None

            plt.plot(
                range(len(metric_hist)-1),
                metric_hist[1:],
                label="Task {} - {}".format(task, metric_name),
                linestyle=linestyles[metric.split("_")[0]],
                color=color
            )

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("{} vs Epochs".format(" and ".join(clean_names)))
    plt.legend()

    if(outfile):
        plt.savefig(outfile)

    if(show):
        plt.show()

    plt.close()
        

def bar_chart(results, task_indexes, vs_paper=True, outfile=None, show=True):

    # Distinct colors
    name = "tab20"
    cmap = get_cmap(name)
    colors = list(cmap.colors)

    if(not vs_paper):
        np.random.shuffle(colors)
        colors = tuple(colors)

    y0 = []
    y1 = []

    barWidth = 0.7
    if(vs_paper):
        barWidth /= 2

    for task in task_indexes:
        y0.append(results[task]["eval_accuracy"]/100)
        y1.append(paper_accuracy[task]/100)

    r0 = np.arange(len(y0))
    r0 += 1
    r1 = [x + barWidth for x in r0]

    if(vs_paper):
        plt.bar(r1, y1, tick_label=r1, color=colors[7], width=barWidth, align='center', edgecolor="white", label="Paper Accuracy")


    plt.bar(r0, y0, tick_label=r0, color=colors[0] if vs_paper else colors, width=barWidth, align='center', edgecolor="white", label="Evaluation Accuracy")


    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.title("Evaluation Accuracy By Task")
    plt.legend(loc='center left', bbox_to_anchor=(0.73, 1.1))

    if(outfile):
        plt.savefig(outfile)

    if(show):
        plt.show() 

    plt.close()


paper_accuracy = {
    1: 100,
    2: 98.2,
    3: 95.2,
    4: 100,
    5: 99.3,
    6: 100,
    7: 96.9,
    8: 96.5,
    9: 100,
    10: 97.5,
    11: 99.9,
    12: 100,
    13: 99.8,
    14: 100,
    15: 100,
    16: 99.4,
    17: 59.6,
    18: 95.3,
    19: 34.5,
    20: 100,
}

r = process_train_log(LOGPATH)

def gen_all_plots(outpath=None, show=True):

    r = process_train_log(LOGPATH)

    outpath = outpath or "./plots"
    if(not os.path.exists(outpath)):
       os.mkdir(outpath)

    bar_chart(r, list(range(1, 1+20)), True, show=show, \
              outfile=os.path.join(outpath,"task_accuracy_vs_paper.png"))


    bar_chart(r, list(range(1, 1+20)), False, show=show, \
              outfile=os.path.join(outpath,"task_accuracy.png"))
    

    for x in [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15], [16,17,18,19,20]]:
        plot_tasks(r, x, "Task {} - {}: Loss vs Epochs".format(min(x), max(x)),
                   show=show,
                   outfile=os.path.join(outpath, "task{}_{}_loss.png".format(min(x), max(x))))

    for x in range(1, 1+20):
        plot_tasks(r, [x], "Task {}: Loss vs Epochs".format(x),
                  show=False,
                  outfile=os.path.join(outpath, "task{}_loss.png".format(x)))



for k in sorted(r.keys()):
    print(k, len(r[k]["train_loss"]), r[k]["train_loss"][-1], r[k]["eval_loss"][-1], r[k]["eval_accuracy"])

    
# gen_all_plots()

#q = plot_tasks(r, list(range(1, 1+5)), "Task 1: Training and Evaluation Loss vs Epochs", outfile="Task 1.png")
#q = bar_chart(r, list(range(1, 1+20)), not False)




##for k in r.keys():
##	print(k, len(r[k]["test_loss"]), len(r[k]["eval_loss"]))
