#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

def load_data(data_folder):
    # Read the CSV files
    fnames = ['train_FD001', 'train_FD002', 'train_FD003', 'train_FD004']
    cols = ['machine', 'cycle', 'p1', 'p2', 'p3'] + [f's{i}' for i in range(1, 22)]
    datalist = []
    nmcn = 0
    for fstem in fnames:
        # Read data
        dfname = os.path.join(f'{data_folder}', f'{fstem}.txt')
        data = pd.read_csv(dfname, sep=' ', header=None)
        # Drop the last two columns (parsing errors)
        data.drop(columns=[26, 27], inplace=True)
        # Replace column names
        data.columns = cols
        # Add the data source
        data['src'] = fstem
        # Shift the machine numbers
        data['machine'] += nmcn
        nmcn += len(data['machine'].unique())
        # Generate RUL data
        cnts = data.groupby('machine')[['cycle']].count()
        cnts.columns = ['ftime']
        data = data.join(cnts, on='machine')
        data['rul'] = data['ftime'] - data['cycle']
        data.drop(columns=['ftime'], inplace=True)
        # Store in the list
        datalist.append(data)
    # Concatenate
    data = pd.concat(datalist)
    # Put the 'src' field at the beginning and keep 'rul' at the end
    data = data[['src'] + cols + ['rul']]
    # data.columns = cols
    return data


def split_by_field(data, field):
    res = {}
    for fval, gdata in data.groupby(field):
        res[fval] = gdata
    return res


def plot_dataframe(data, labels=None, vmin=-1.96, vmax=1.96,
        figsize=None, s=4):
    plt.figure(figsize=figsize)
    plt.imshow(data.T.iloc[:, :], aspect='auto',
            cmap='RdBu', vmin=vmin, vmax=vmax)
    if labels is not None:
        # nonzero = data.index[labels != 0]
        ncol = len(data.columns)
        lvl = - 0.05 * ncol
        # plt.scatter(nonzero, lvl*np.ones(len(nonzero)),
        #         s=s, color='tab:orange')
        plt.scatter(labels.index, np.ones(len(labels)) * lvl,
                s=s,
                color=plt.get_cmap('tab10')(np.mod(labels, 10)))
    plt.tight_layout()


def plot_series(data, labels=None,
                    windows=None,
                    predictions=None,
                    highlights=None,
                    val_start=None,
                    test_start=None,
                    figsize=None,
                    show_sampling_points=False,
                    show_markers=False,
                    filled_version=None):
    # Open a new figure
    plt.figure(figsize=figsize)
    # Plot data
    if not show_markers:
        plt.plot(data.index, data.values, zorder=0)
    else:
        plt.plot(data.index, data.values, zorder=0,
                marker='.', markersize=3)
    if filled_version is not None:
        filled = filled_version.copy()
        filled[~data['value'].isnull()] = np.nan
        plt.scatter(filled.index, filled,
                marker='.', c='tab:orange', s=5);
    if show_sampling_points:
        vmin = data.min()
        lvl = np.full(len(data.index), vmin)
        plt.scatter(data.index, lvl, marker='.',
                c='tab:red', s=5)
    # Rotated x ticks
    plt.xticks(rotation=45)
    # Plot labels
    if labels is not None:
        plt.scatter(labels.values, data.loc[labels],
                    color=anomaly_color, zorder=2, s=5)
    # Plot windows
    if windows is not None:
        for _, wdw in windows.iterrows():
            plt.axvspan(wdw['begin'], wdw['end'],
                        color=anomaly_color, alpha=0.3, zorder=1)
    # Plot training data
    if val_start is not None:
        plt.axvspan(data.index[0], val_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is None and test_start is not None:
        plt.axvspan(data.index[0], test_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is not None:
        plt.axvspan(val_start, test_start,
                    color=validation_color, alpha=0.1, zorder=-1)
    if test_start is not None:
        plt.axvspan(test_start, data.index[-1],
                    color=test_color, alpha=0.3, zorder=0)
    # Predictions
    if predictions is not None:
        plt.scatter(predictions.values, data.loc[predictions],
                    color=prediction_color, alpha=.4, zorder=3,
                    s=5)
    plt.grid(linestyle=':')
    plt.tight_layout()



def partition_by_machine(data, tr_machines):
    # Separate
    tr_machines = set(tr_machines)
    tr_list, ts_list = [], []
    for mcn, gdata in data.groupby('machine'):
        if mcn in tr_machines:
            tr_list.append(gdata)
        else:
            ts_list.append(gdata)
    # Collate again
    tr_data = pd.concat(tr_list)
    ts_data = pd.concat(ts_list)
    return tr_data, ts_data


def build_nn_model(input_shape, output_shape, hidden, output_activation='linear'):
    model_in = keras.Input(shape=input_shape, dtype='float32')
    x = model_in
    for h in hidden:
        x = layers.Dense(h, activation='relu')(x)
    model_out = layers.Dense(output_shape, activation=output_activation)(x)
    model = keras.Model(model_in, model_out)
    return model


def plot_nn_model(model):
    return keras.utils.plot_model(model, show_shapes=True,
            show_layer_names=True, rankdir='LR')


def plot_training_history(history=None,
        figsize=None, print_final_scores=True):
    plt.figure(figsize=figsize)
    for metric in history.history.keys():
        plt.plot(history.history[metric], label=metric)
    # if 'val_loss' in history.history.keys():
    #     plt.plot(history.history['val_loss'], label='val. loss')
    if len(history.history.keys()) > 0:
        plt.legend()
    plt.xlabel('epochs')
    plt.grid(linestyle=':')
    plt.tight_layout()
    plt.show()
    if print_final_scores:
        trl = history.history["loss"][-1]
        s = f'Final loss: {trl:.4f} (training)'
        if 'val_loss' in history.history:
            vll = history.history["val_loss"][-1]
            s += f', {vll:.4f} (validation)'
        print(s)


# from keras_tqdm import TQDMNotebookCallback

# def train_nn_model(model, X, y, loss, epochs=20,
#         verbose=0, patience=10, batch_size=32,
#         validation_split=0.0, **fit_params):
def train_nn_model(model, X, y, loss,
        verbose=0, patience=10,
        validation_split=0.0, **fit_params):
    # Compile the model
    model.compile(optimizer='Adam', loss=loss)
    # Build the early stop callback
    cb = []
    if validation_split > 0:
        cb += [callbacks.EarlyStopping(patience=patience,
            restore_best_weights=True)]
    # Train the model
    history = model.fit(X, y, callbacks=cb,
            validation_split=validation_split,
            verbose=verbose, **fit_params)
    return history


def plot_pred_scatter(y_pred, y_true, figsize=None):
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, y_true, marker='.', alpha=0.1)
    xl, xu = plt.xlim()
    yl, yu = plt.ylim()
    l, u = min(xl, yl), max(xu, yu)
    plt.plot([l, u], [l, u], ':', c='0.3')
    plt.xlim(l, u)
    plt.ylim(l, u)
    plt.xlabel('prediction')
    plt.ylabel('target')
    plt.grid(linestyle=':')
    plt.tight_layout()


def plot_rul(pred=None, target=None,
        stddev=None,
        q1_3=None,
        same_scale=True,
        figsize=None):
    plt.figure(figsize=figsize)
    if target is not None:
        plt.plot(range(len(target)), target, label='target',
                color='tab:orange')
    if pred is not None:
        if same_scale or target is None:
            ax = plt.gca()
        else:
            ax = plt.gca().twinx()
        ax.plot(range(len(pred)), pred, label='pred',
                color='tab:blue')
        if stddev is not None:
            ax.fill_between(range(len(pred)),
                    pred-stddev, pred+stddev,
                    alpha=0.3, color='tab:blue', label='+/- std')
        if q1_3 is not None:
            ax.fill_between(range(len(pred)),
                    q1_3[0], q1_3[1],
                    alpha=0.3, color='tab:blue', label='1st/3rd quartile')
    plt.legend()
    plt.grid(linestyle=':')
    plt.tight_layout()


class RULCostModel:
    def __init__(self, maintenance_cost, safe_interval=0):
        self.maintenance_cost = maintenance_cost
        self.safe_interval = safe_interval

    def cost(self, machine, pred, thr, return_margin=False):
        # Merge machine and prediction data
        tmp = np.array([machine, pred]).T
        tmp = pd.DataFrame(data=tmp,
                           columns=['machine', 'pred'])
        # Cost computation
        cost = 0
        nfails = 0
        slack = 0
        for mcn, gtmp in tmp.groupby('machine'):
            idx = np.nonzero(gtmp['pred'].values < thr)[0]
            if len(idx) == 0:
                cost += self.maintenance_cost
                nfails += 1
            else:
                cost -= max(0, idx[0] - self.safe_interval)
                slack += len(gtmp) - idx[0]
        if not return_margin:
            return cost
        else:
            return cost, nfails, slack


def opt_threshold_and_plot(machine, pred, th_range, cmodel,
        plot=True, figsize=None):
    # Compute the optimal threshold
    costs = [cmodel.cost(machine, pred, thr) for thr in th_range]
    opt_th = th_range[np.argmin(costs)]
    # Plot
    if plot:
        plt.figure(figsize=figsize)
        plt.plot(th_range, costs)
        plt.grid(linestyle=':')
        plt.tight_layout()
    # Return the threshold
    return opt_th



def sliding_window_2D(data, wlen, stride=1):
    # Get shifted tables
    m = len(data)
    lt = [data.iloc[i:m-wlen+i+1:stride, :].values for i in range(wlen)]
    # Reshape to add a new axis
    s = lt[0].shape
    for i in range(wlen):
        lt[i] = lt[i].reshape(s[0], 1, s[1])
    # Concatenate
    wdata = np.concatenate(lt, axis=1)
    return wdata


def sliding_window_by_machine(data, wlen, cols, stride=1):
    l_w, l_m, l_r = [], [], []
    for mcn, gdata in data.groupby('machine'):
        # Apply a sliding window
        tmp_w = sliding_window_2D(gdata[cols], wlen, stride)
        # Build the machine vector
        tmp_m = gdata['machine'].iloc[wlen-1::stride]
        # Build the RUL vector
        tmp_r = gdata['rul'].iloc[wlen-1::stride]
        # Store everything
        l_w.append(tmp_w)
        l_m.append(tmp_m)
        l_r.append(tmp_r)
    res_w = np.concatenate(l_w)
    res_m = np.concatenate(l_m)
    res_r = np.concatenate(l_r)
    return res_w, res_m, res_r


def build_cnn_model(input_shape, output_shape, wlen, conv=[], hidden=[], output_activation='linear'):
    model_in = keras.Input(shape=input_shape, dtype='float32')
    x = model_in
    for k in conv:
        x = layers.Conv1D(k, kernel_size=3, activation='relu')(x)
    x = layers.Flatten()(x)
    for k in hidden:
        x = layers.Dense(k, activation='relu')(x)
    x = layers.Dense(output_shape, activation=output_activation)(x)
    model = keras.Model(model_in, x)
    return model


def plot_gp(target=None, pred=None, std=None, samples=None,
        target_samples=None, figsize=None):
    plt.figure(figsize=figsize)
    if target is not None:
        plt.plot(target.index, target, c='black', label='target')
    if pred is not None:
        plt.plot(pred.index, pred, c='tab:blue',
                label='predictions')
    if std is not None:
        plt.fill_between(pred.index, pred-1.96*std, pred+1.96*std,
                alpha=.3, fc='tab:blue', ec='None',
                label='95% C.I.')
    # Add scatter plots
    if samples is not None:
        try:
            x = samples.index
            y = samples.values
        except AttributeError:
            x = samples[0]
            y = samples[1]
        plt.scatter(x, y, color='tab:orange',
              label='samples', marker='x')
    if target_samples is not None:
        try:
            x = target_samples.index
            y = target_samples.values
        except AttributeError:
            x = target_samples[0]
            y = target_samples[1]
        plt.scatter(x, y,
                color='black', label='target', s=5)
    plt.legend()
    plt.grid(linestyle=':')
    plt.tight_layout()


class ClassifierCost:

    def __init__(self, machines, X, y, cost_model, init_epochs=20, inc_epochs=3):
        self.machines = machines
        self.X = X
        self.y = y
        self.cost_model = cost_model
        self.nn = build_nn_model(input_shape=(X.shape[1],), output_shape=1,
                hidden=[32, 32], output_activation='sigmoid')
        self.init_epochs = init_epochs
        self.inc_epochs = inc_epochs

        self.nmc = len(machines.unique())
        self.stored_weights = {}
        self.is_init = False

    # def __call__(self, params):
    #     theta = params[0]
    #     print(f'theta: {theta:.2f}, ', end='')
    #     # Redefine classes
    #     lbl = (self.y >= theta)
    #     # Determine the number of epochs
    #     epochs = self.init_epochs if not self.is_init else self.inc_epochs
    #     self.is_init = True
    #     # Retrain
    #     train_nn_model(self.nn, self.X, lbl, loss='binary_crossentropy', epochs=epochs,
    #             verbose=0, patience=10, batch_size=32, validation_split=0.2)
    #     # Store weights
    #     self.stored_weights[theta] = self.nn.get_weights()
    #     # Evaluate cost
    #     pred = np.round(self.nn.predict(self.X, verbose=0).ravel())
    #     cost, fails, slack = self.cost_model.cost(self.machines, pred, 0.5, return_margin=True)
    #     print(f'avg. cost: {cost/self.nmc:.2f}, avg. fails: {fails/self.nmc:.2f}, avg. slack: {slack/self.nmc:.2f}')
    #     return cost

    def __call__(self, eps, verbose=0):
        if verbose > 0:
            print(f'eps: {eps:.2f}, ', end='')
        # Redefine classes
        lbl = (self.y >= eps)
        # Determine the number of epochs
        epochs = self.init_epochs if not self.is_init else self.inc_epochs
        self.is_init = True
        # Retrain
        train_nn_model(self.nn, self.X, lbl, loss='binary_crossentropy', epochs=epochs,
                verbose=0, patience=10, batch_size=32, validation_split=0.2)
        # Store weights
        self.stored_weights[eps] = self.nn.get_weights()
        # Evaluate cost
        pred = np.round(self.nn.predict(self.X, verbose=0).ravel())
        cost, fails, slack = self.cost_model.cost(self.machines, pred, 0.5, return_margin=True)
        if verbose > 0:
            print(f'avg. cost: {cost/self.nmc:.2f}, avg. fails: {fails/self.nmc:.2f}, avg. slack: {slack/self.nmc:.2f}')
        return -cost
