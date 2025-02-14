import os
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class FoldMetrics:
    """
    Records metrics for one fold.
    Stores a list of (epoch, train_loss) tuples and a dictionary for test metrics.
    """
    def __init__(self, fold_number):
        self.fold_number = fold_number
        self.train_losses = []  # list of tuples: (epoch, loss)
        self.test_metrics = {}  # dict: epoch -> metrics dict

    def add_train_loss(self, epoch, loss):
        self.train_losses.append((epoch, loss))

    def add_test_metrics(self, epoch, metrics):
        self.test_metrics[epoch] = metrics

    def __str__(self):
        s = f"Fold {self.fold_number} Metrics:\n"
        s += "Epoch\tTrainLoss\tTestLoss\tHR@5\tNDCG@5\tHR@10\tNDCG@10\n"
        for epoch in sorted(self.test_metrics.keys()):
            train_loss = next((tl for ep, tl in self.train_losses if ep == epoch), None)
            test = self.test_metrics.get(epoch, None)
            test_loss = test['loss'] if test is not None else float('nan')
            s += (f"{epoch}\t{train_loss:.4f}\t{test_loss:.4f}\t"
                  f"{test['HR5']:.4f}\t{test['NDCG5']:.4f}\t{test['HR10']:.4f}\t{test['NDCG10']:.4f}\n")
        return s

class AverageMetrics:
    """
    Averages the metrics from multiple folds.
    """
    def __init__(self):
        self.avg_train_loss = {}
        self.avg_test_metrics = {}
        self.num_folds = 0

    def add_fold_metrics(self, fold_metric):
        self.num_folds += 1
        for epoch, loss in fold_metric.train_losses:
            self.avg_train_loss.setdefault(epoch, []).append(loss)
        for epoch, metrics in fold_metric.test_metrics.items():
            self.avg_test_metrics.setdefault(epoch, []).append(metrics)

    def compute_averages(self):
        self.avg_train_loss = {epoch: np.mean(losses) for epoch, losses in self.avg_train_loss.items()}

        def avg_dict(metrics_list):
            avg_d = {}
            for key in metrics_list[0].keys():
                avg_d[key] = np.mean([m[key] for m in metrics_list])
            return avg_d

        self.avg_test_metrics = {epoch: avg_dict(metrics_list) for epoch, metrics_list in self.avg_test_metrics.items()}

    def __str__(self):
        s = "Average Metrics Across Folds:\n"
        s += "Epoch\tAvgTrainLoss\tAvgTestLoss\tAvgHR@5\tAvgNDCG@5\tAvgHR@10\tAvgNDCG@10\n"
        for epoch in sorted(self.avg_test_metrics.keys()):
            train_loss = self.avg_train_loss.get(epoch, None)
            test = self.avg_test_metrics.get(epoch, None)
            test_loss = test['loss'] if test is not None else float('nan')
            s += (f"{epoch}\t{train_loss:.4f}\t{test_loss:.4f}\t"
                  f"{test['HR5']:.4f}\t{test['NDCG5']:.4f}\t{test['HR10']:.4f}\t{test['NDCG10']:.4f}\n")
        return s

class LossRecorder:
    """
    Records per-epoch losses for training and test over folds.
    Can save to and load from a JSON file and plot the average curves.
    """
    def __init__(self, save_dir=None):
        self.fold_losses = {}  # key: fold number
        self.save_dir = save_dir if save_dir is not None else "."

    def add_fold(self, fold, train_losses, test_losses):
        self.fold_losses[fold] = {
            'train': train_losses,
            'test': test_losses
        }

    def compute_average_losses(self):
        all_train = defaultdict(list)
        for data in self.fold_losses.values():
            for epoch, loss in enumerate(data.get('train', []), start=1):
                all_train[epoch].append(loss)
        avg_train = {epoch: np.mean(losses) for epoch, losses in all_train.items()}

        all_test = defaultdict(list)
        for data in self.fold_losses.values():
            for i, loss in enumerate(data.get('test', [])):
                epoch = (i + 1) * 10
                all_test[epoch].append(loss)
        avg_test = {epoch: np.mean(losses) for epoch, losses in all_test.items()}

        return avg_train, avg_test

    def save_to_file(self, filename=None):
        if filename is None:
            filename = os.path.join(self.save_dir, "loss_data.json")
        serializable = {str(fold): self.fold_losses[fold] for fold in self.fold_losses}
        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Loss data saved to {filename}")

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        lr_obj = cls()
        lr_obj.fold_losses = {int(fold): data_fold for fold, data_fold in data.items()}
        return lr_obj

    def plot_losses(self):
        avg_train, avg_test = self.compute_average_losses()
        epochs_train = sorted(avg_train.keys())
        epochs_test = sorted(avg_test.keys())
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_train, [avg_train[e] for e in epochs_train],
                 color='blue', linestyle='-', label='Train Loss')
        plt.plot(epochs_test, [avg_test[e] for e in epochs_test],
                 color='red', linestyle='-', label='Test Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Average Loss per Epoch (10-Fold CV)")
        plt.legend()
        plt.grid(True)
        plt.show()

class MetricsRecorder:
    """
    Records test metrics (HR@5, HR@10, NDCG@5, NDCG@10) from each fold at evaluation epochs.
    """
    def __init__(self, save_dir=None):
        self.metrics = defaultdict(list)
        self.save_dir = save_dir if save_dir is not None else "."

    def add_fold(self, fold_metric):
        for epoch, met in fold_metric.test_metrics.items():
            self.metrics[epoch].append(met)

    def compute_average(self):
        avg = {}
        for epoch, met_list in self.metrics.items():
            avg[epoch] = {k: np.mean([m[k] for m in met_list]) for k in met_list[0].keys()}
        return avg

    def save_to_file(self, filename=None):
        if filename is None:
            filename = os.path.join(self.save_dir, "average_test_metrics.txt")
        avg = self.compute_average()
        with open(filename, 'w') as f:
            f.write("Epoch\tHR@5\tNDCG@5\tHR@10\tNDCG@10\n")
            for epoch in sorted(avg.keys()):
                m = avg[epoch]
                f.write(f"{epoch}\t{m['HR5']:.4f}\t{m['NDCG5']:.4f}\t{m['HR10']:.4f}\t{m['NDCG10']:.4f}\n")
        print(f"Average test metrics saved to {filename}")

    def plot_metrics(self):
        avg = self.compute_average()
        epochs = sorted(avg.keys())
        hr5 = [avg[e]['HR5'] for e in epochs]
        ndcg5 = [avg[e]['NDCG5'] for e in epochs]
        hr10 = [avg[e]['HR10'] for e in epochs]
        ndcg10 = [avg[e]['NDCG10'] for e in epochs]
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, hr5, color='blue', linestyle='-', label='HR@5')
        plt.plot(epochs, hr10, color='red', linestyle='-', label='HR@10')
        plt.plot(epochs, ndcg5, color='green', linestyle='-', label='NDCG@5')
        plt.plot(epochs, ndcg10, color='purple', linestyle='-', label='NDCG@10')
        plt.xlabel("Evaluation Epoch")
        plt.ylabel("Metric Value")
        plt.title("Average Test Metrics (10-Fold CV)")
        plt.legend()
        plt.grid(True)
        plt.show()

class TuningRecorder:
    """
    Records tuning metrics (HR@10) for each candidate parameter over evaluation epochs.
    """
    def __init__(self, parameter_name, candidates=None, save_dir=None):
        self.parameter_name = parameter_name
        self.candidates = candidates if candidates is not None else []
        self.data = defaultdict(lambda: defaultdict(list))
        self.save_dir = save_dir if save_dir is not None else "."

    def record(self, candidate, eval_epoch, hr10):
        self.data[candidate][eval_epoch].append(hr10)

    def compute_average(self):
        avg_data = {}
        for candidate, d in self.data.items():
            avg_data[candidate] = {epoch: np.mean(hr_list) for epoch, hr_list in d.items()}
        return avg_data

    def save_to_file(self, filename=None):
        if filename is None:
            filename = os.path.join(self.save_dir, f"tuning_{self.parameter_name}.json")
        avg_data = self.compute_average()
        serializable = {str(candidate): {str(epoch): float(hr) for epoch, hr in avg_dict.items()}
                        for candidate, avg_dict in avg_data.items()}
        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Tuning data for {self.parameter_name} saved to {filename}")

    @classmethod
    def load_from_file(cls, parameter_name, filename, save_dir=None):
        with open(filename, 'r') as f:
            data = json.load(f)
        tr = cls(parameter_name, save_dir=save_dir)
        for candidate, epoch_dict in data.items():
            candidate_val = float(candidate) if '.' in candidate else int(candidate)
            for epoch, hr in epoch_dict.items():
                epoch_val = int(epoch)
                tr.data[candidate_val][epoch_val].append(hr)
        return tr

    def plot(self):
        avg_data = self.compute_average()
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10.colors
        for i, (candidate, epoch_dict) in enumerate(sorted(avg_data.items())):
            epochs = sorted(epoch_dict.keys())
            hr_values = [epoch_dict[e] for e in epochs]
            plt.plot(epochs, hr_values, color=colors[i % len(colors)], linestyle='-',
                     label=f"{self.parameter_name}={candidate}")
        plt.xlabel("Evaluation Epoch")
        plt.ylabel("Average HR@10")
        plt.title(f"Tuning Results for {self.parameter_name}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def find_best(self):
        """
        Finds the candidate with the highest average HR@10 at evaluation epoch 100.
        If epoch 100 is not available for a candidate, it uses the maximum average among its epochs.
        """
        avg_data = self.compute_average()
        best = -np.inf
        best_candidate = None
        for candidate, epoch_dict in avg_data.items():
            if 100 in epoch_dict:
                avg_final = epoch_dict[100]
            else:
                avg_final = max(epoch_dict.values()) if epoch_dict else -np.inf
            if avg_final > best:
                best = avg_final
                best_candidate = candidate
        return best_candidate
