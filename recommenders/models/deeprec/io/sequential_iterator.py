# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
import random
import os
from recommenders.models.deeprec.io.iterator import BaseIterator
from recommenders.models.deeprec.deeprec_utils import load_dict


__all__ = ["SequentialIterator"]


class SequentialIterator(BaseIterator):
    def __init__(self, hparams, graph, col_spliter="\t"):
        """Initialize an iterator. Create necessary placeholders for the model.

        Args:
            hparams (object): Global hyper-parameters. Some key settings such as #_feature and #_field are there.
            graph (object): The running graph. All created placeholder will be added to this graph.
            col_spliter (str): Column splitter in one line.
        """
        self.col_spliter = col_spliter
        self.hparams = hparams
        user_vocab, item_vocab, cate_vocab = (
            hparams.user_vocab,
            hparams.item_vocab,
            hparams.cate_vocab
        )
        self.userdict, self.itemdict, self.catedict = (
            load_dict(user_vocab),
            load_dict(item_vocab),
            load_dict(cate_vocab),
        )

        self.max_seq_length = hparams.max_seq_length
        self.batch_size = hparams.batch_size
        self.iter_data = dict()

        self.graph = graph
        with self.graph.as_default():
            self.labels = tf.compat.v1.placeholder(tf.float32, [None, 1], name="label")
            self.users = tf.compat.v1.placeholder(tf.int32, [None], name="users")
            self.items = tf.compat.v1.placeholder(tf.int32, [None], name="items")
            self.cates = tf.compat.v1.placeholder(tf.int32, [None], name="cates")
            self.item_history = tf.compat.v1.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_history"
            )
            self.item_cate_history = tf.compat.v1.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_history"
            )
            self.mask = tf.compat.v1.placeholder(
                tf.int32, [None, self.max_seq_length], name="mask"
            )
            self.time = tf.compat.v1.placeholder(tf.float32, [None], name="time")
            self.time_diff = tf.compat.v1.placeholder(
                tf.float32, [None, self.max_seq_length], name="time_diff"
            )
            self.time_from_first_action = tf.compat.v1.placeholder(
                tf.float32, [None, self.max_seq_length], name="time_from_first_action"
            )
            self.time_to_now = tf.compat.v1.placeholder(
                tf.float32, [None, self.max_seq_length], name="time_to_now"
            )
            self.cont_feat = tf.compat.v1.placeholder(
                tf.float32, [None, self.hparams.cont_feat_len], name="continuous_features"
            )
    def _get_file_line_cnt(self, input_file):
        idx = 0
        with open(input_file, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                idx += 1
        return idx

    def parse_file(self, input_file):
        """Parse the file to A list ready to be used for downstream tasks.

        Args:
            input_file: One of train, valid or test file which has never been parsed.

        Returns:
            list: A list with parsing result.
        """
        print('parse file {0}'.format(os.path.basename(input_file)))
        line_cnt = self._get_file_line_cnt(input_file)
        print('line cnt = {0}'.format(line_cnt))
        res = np.empty(line_cnt, dtype=object)
        idx = 0
        with open(input_file, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break  
                if idx % 10000 == 0:
                    print('\rparsing line {0}k'.format(idx // 1000), end=' ')
          
                res[idx] = self.parser_one_line(line)
                idx += 1 
        print('finish parse file {0}'.format(os.path.basename(input_file)))

        return res

    def parser_one_line(self, line):
        """Parse one string line into feature values.

        Args:
            line (str): a string indicating one instance.
                This string contains tab-separated values including:
                label, user_hash, item_hash, item_cate, operation_time, item_history_sequence,
                item_cate_history_sequence, and time_history_sequence.

        Returns:
            list: Parsed results including `label`, `user_id`, `item_id`, `item_cate`, `item_history_sequence`, `cate_history_sequence`,
            `current_time`, `time_diff`, `time_from_first_action`, `time_to_now`.

        """
        words = line.strip().split(self.col_spliter)
        label = int(words[0])
        user_id = self.userdict[words[1]] if words[1] in self.userdict else 0
        item_id = self.itemdict[words[2]] if words[2] in self.itemdict else 0
        item_cate = self.catedict[words[3]] if words[3] in self.catedict else 0
        current_time = float(words[4])

        item_history_sequence = []
        cate_history_sequence = []
        time_history_sequence = []

        item_history_words = words[5].strip().split(",")
        for item in item_history_words:
            item_history_sequence.append(
                self.itemdict[item] if item in self.itemdict else 0
            )
        item_history_sequence = np.array(item_history_sequence, dtype=np.int)

        cate_history_words = words[6].strip().split(",")
        for cate in cate_history_words:
            cate_history_sequence.append(
                self.catedict[cate] if cate in self.catedict else 0
            )
        cate_history_sequence = np.array(cate_history_sequence, dtype=np.int)

        time_history_words = words[7].strip().split(",")
        time_history_sequence = [float(i) for i in time_history_words]
        time_history_sequence = np.array(time_history_sequence, dtype=np.float32)

        if self.hparams.feat_start_idx >=0 :
            cont_feats = np.array([float(a) for a in words[8].split(",")[self.hparams.feat_start_idx:self.hparams.feat_end_idx]], dtype=np.float32) if self.hparams.use_cont_feat else None
        else:
            cont_feats = np.array([float(a) for a in words[8].split(",")], dtype=np.float32) if self.hparams.use_cont_feat else None


        time_range = 3600 * 24

        time_diff = []
        for i in range(len(time_history_sequence) - 1):
            diff = (
                time_history_sequence[i + 1] - time_history_sequence[i]
            ) / time_range
            diff = max(diff, 0.5)
            time_diff.append(diff)
        last_diff = (current_time - time_history_sequence[-1]) / time_range
        last_diff = max(last_diff, 0.5)
        time_diff.append(last_diff)
        time_diff = np.array(time_diff, dtype=np.float32)
        time_diff = np.log(time_diff)

        time_from_first_action = []
        first_time = time_history_sequence[0]
        time_from_first_action = [
            (t - first_time) / time_range for t in time_history_sequence[1:]
        ]
        time_from_first_action = [max(t, 0.5) for t in time_from_first_action]
        last_diff = (current_time - first_time) / time_range
        last_diff = max(last_diff, 0.5)
        time_from_first_action.append(last_diff)
        time_from_first_action = np.array(time_from_first_action, dtype=np.float32)
        time_from_first_action = np.log(time_from_first_action)

        time_to_now = []
        time_to_now = [(current_time - t) / time_range for t in time_history_sequence]
        time_to_now = [max(t, 0.5) for t in time_to_now]
        time_to_now = np.array(time_to_now, dtype=np.float32)
        time_to_now = np.log(time_to_now)

        return (
            label,
            user_id,
            item_id,
            item_cate,
            item_history_sequence,
            cate_history_sequence,
            current_time,
            time_diff,
            time_from_first_action,
            time_to_now,
            cont_feats
        )  

    def load_data_from_file_training(self, infile, batch_num_ngs=0, min_seq_length=1):
        """Read and parse data from a file.

        Args:
            infile (str): Text input file. Each line in this file is an instance.
            batch_num_ngs (int): The number of negative sampling here in batch.
                0 represents that there is no need to do negative sampling here.
            min_seq_length (int): The minimum number of a sequence length.
                Sequences with length lower than min_seq_length will be ignored.

        Yields:
            object: An iterator that yields parsed results, in the format of graph `feed_dict`.
        """
        label_list = []
        user_list = []
        item_list = []
        item_cate_list = []
        item_history_batch = []
        item_cate_history_batch = []
        time_list = []
        time_diff_list = []
        time_from_first_action_list = []
        time_to_now_list = []
        cont_feat_list = []

        cnt = 0

        if infile not in self.iter_data:
            if ';' not in infile:
                lines = self.parse_file(infile)
                if self.hparams.shuffle:
                    np.random.shuffle(lines)
                    print('shuffle lines finished.')
                self.iter_data[infile] = lines
            else:
                print('merging multiple train files...')
                content_list = [] 
                for each_file in infile.split(';'):
                    lines = self.parse_file(each_file)
                    if len(content_list) > 0:
                        lines = np.repeat(lines, 10, axis=0)
                    content_list.append(lines)
                lines = np.concatenate(content_list, axis=0)
                if self.hparams.shuffle:
                    np.random.shuffle(lines)
                    print('shuffle lines finished.')
                self.iter_data[infile] = lines
                    
        else:
            lines = self.iter_data[infile]

        if batch_num_ngs > 0:
            random.shuffle(lines)

        ptr, N = -1, len(lines)
        while True:
            ptr += 1
            if ptr >= N:
                ptr = 0

            (
                label,
                user_id,
                item_id,
                item_cate,
                item_history_sequence,
                item_cate_history_sequence,
                current_time,
                time_diff,
                time_from_first_action,
                time_to_now,
                cont_feats
            ) = lines[ptr]

            if len(item_history_sequence) < min_seq_length:
                continue

            label_list.append(label)
            user_list.append(user_id)
            item_list.append(item_id)
            item_cate_list.append(item_cate)
            item_history_batch.append(item_history_sequence)
            item_cate_history_batch.append(item_cate_history_sequence)
            time_list.append(current_time)
            time_diff_list.append(time_diff)
            time_from_first_action_list.append(time_from_first_action)
            time_to_now_list.append(time_to_now)
            cont_feat_list.append(cont_feats)

            cnt += 1
            if cnt == self.batch_size:
                res = self._convert_data(
                    label_list,
                    user_list,
                    item_list,
                    item_cate_list,
                    item_history_batch,
                    item_cate_history_batch,
                    time_list,
                    time_diff_list,
                    time_from_first_action_list,
                    time_to_now_list,
                    batch_num_ngs,
                    cont_feat_list
                )
                batch_input = self.gen_feed_dict(res)
                yield batch_input if batch_input else None
                label_list = []
                user_list = []
                item_list = []
                item_cate_list = []
                item_history_batch = []
                item_cate_history_batch = []
                time_list = []
                time_diff_list = []
                time_from_first_action_list = []
                time_to_now_list = []
                cont_feat_list = []
                cnt = 0
 


    def load_data_from_file(self, infile, batch_num_ngs=0, min_seq_length=1):
        """Read and parse data from a file.

        Args:
            infile (str): Text input file. Each line in this file is an instance.
            batch_num_ngs (int): The number of negative sampling here in batch.
                0 represents that there is no need to do negative sampling here.
            min_seq_length (int): The minimum number of a sequence length.
                Sequences with length lower than min_seq_length will be ignored.

        Yields:
            object: An iterator that yields parsed results, in the format of graph `feed_dict`.
        """
        label_list = []
        user_list = []
        item_list = []
        item_cate_list = []
        item_history_batch = []
        item_cate_history_batch = []
        time_list = []
        time_diff_list = []
        time_from_first_action_list = []
        time_to_now_list = []
        cont_feat_list = []

        cnt = 0

        if infile not in self.iter_data:
            lines = self.parse_file(infile)
            self.iter_data[infile] = lines
        else:
            lines = self.iter_data[infile]

        if batch_num_ngs > 0:
            random.shuffle(lines)

        for line in lines:
            if not line:
                continue

            (
                label,
                user_id,
                item_id,
                item_cate,
                item_history_sequence,
                item_cate_history_sequence,
                current_time,
                time_diff,
                time_from_first_action,
                time_to_now,
                cont_feats
            ) = line
            if len(item_history_sequence) < min_seq_length:
                continue

            label_list.append(label)
            user_list.append(user_id)
            item_list.append(item_id)
            item_cate_list.append(item_cate)
            item_history_batch.append(item_history_sequence)
            item_cate_history_batch.append(item_cate_history_sequence)
            time_list.append(current_time)
            time_diff_list.append(time_diff)
            time_from_first_action_list.append(time_from_first_action)
            time_to_now_list.append(time_to_now)
            cont_feat_list.append(cont_feats)

            cnt += 1
            if cnt == self.batch_size:
                res = self._convert_data(
                    label_list,
                    user_list,
                    item_list,
                    item_cate_list,
                    item_history_batch,
                    item_cate_history_batch,
                    time_list,
                    time_diff_list,
                    time_from_first_action_list,
                    time_to_now_list,
                    batch_num_ngs,
                    cont_feat_list
                )
                batch_input = self.gen_feed_dict(res)
                yield batch_input if batch_input else None
                label_list = []
                user_list = []
                item_list = []
                item_cate_list = []
                item_history_batch = []
                item_cate_history_batch = []
                time_list = []
                time_diff_list = []
                time_from_first_action_list = []
                time_to_now_list = []
                cont_feat_list = []
                cnt = 0
        if cnt > 0:
            res = self._convert_data(
                label_list,
                user_list,
                item_list,
                item_cate_list,
                item_history_batch,
                item_cate_history_batch,
                time_list,
                time_diff_list,
                time_from_first_action_list,
                time_to_now_list,
                batch_num_ngs,
                cont_feat_list
            )
            batch_input = self.gen_feed_dict(res)
            yield batch_input if batch_input else None

    def _convert_data(
        self,
        label_list,
        user_list,
        item_list,
        item_cate_list,
        item_history_batch,
        item_cate_history_batch,
        time_list,
        time_diff_list,
        time_from_first_action_list,
        time_to_now_list,
        batch_num_ngs,
        cont_feat_list
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            label_list (list): A list of ground-truth labels.
            user_list (list): A list of user indexes.
            item_list (list): A list of item indexes.
            item_cate_list (list): A list of category indexes.
            item_history_batch (list): A list of item history indexes.
            item_cate_history_batch (list): A list of category history indexes.
            time_list (list): A list of current timestamp.
            time_diff_list (list): A list of timestamp between each sequential operations.
            time_from_first_action_list (list): A list of timestamp from the first operation.
            time_to_now_list (list): A list of timestamp to the current time.
            batch_num_ngs (int): The number of negative sampling while training in mini-batch.

        Returns:
            dict: A dictionary, containing multiple numpy arrays that are convenient for further operation.
        """
        # print('\r##INFO: batch_num_ngs = {0}'.format(batch_num_ngs), end='')
        if False: ##batch_num_ngs:
            instance_cnt = len(label_list)
            if instance_cnt < 5:
                return

            label_list_all = []
            item_list_all = []
            item_cate_list_all = []
            user_list_all = np.asarray(
                [[user] * (batch_num_ngs + 1) for user in user_list], dtype=np.int32
            ).flatten()
            time_list_all = np.asarray(
                [[t] * (batch_num_ngs + 1) for t in time_list], dtype=np.float32
            ).flatten()

            history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]
            max_seq_length_batch = self.max_seq_length
            item_history_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("int32")
            item_cate_history_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("int32")
            time_diff_batch = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("float32")
            time_from_first_action_batch = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("float32")
            time_to_now_batch = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("float32")
            mask = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length_batch)
            ).astype("float32")

            for i in range(instance_cnt):
                this_length = min(history_lengths[i], max_seq_length_batch)
                for index in range(batch_num_ngs + 1):
                    item_history_batch_all[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(item_history_batch[i][-this_length:], dtype=np.int32)
                    item_cate_history_batch_all[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(
                        item_cate_history_batch[i][-this_length:], dtype=np.int32
                    )
                    mask[i * (batch_num_ngs + 1) + index, :this_length] = 1.0
                    time_diff_batch[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(time_diff_list[i][-this_length:], dtype=np.float32)
                    time_from_first_action_batch[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(
                        time_from_first_action_list[i][-this_length:], dtype=np.float32
                    )
                    time_to_now_batch[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(time_to_now_list[i][-this_length:], dtype=np.float32)

            for i in range(instance_cnt):
                positive_item = item_list[i]
                label_list_all.append(1)
                item_list_all.append(positive_item)
                item_cate_list_all.append(item_cate_list[i])
                count = 0
                while batch_num_ngs:
                    random_value = random.randint(0, instance_cnt - 1)
                    negative_item = item_list[random_value]
                    if negative_item == positive_item:
                        continue
                    label_list_all.append(0)
                    item_list_all.append(negative_item)
                    item_cate_list_all.append(item_cate_list[random_value])
                    count += 1
                    if count == batch_num_ngs:
                        break

            res = {}
            res["labels"] = np.asarray(label_list_all, dtype=np.float32).reshape(-1, 1)
            res["users"] = user_list_all
            res["items"] = np.asarray(item_list_all, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list_all, dtype=np.int32)
            res["item_history"] = item_history_batch_all
            res["item_cate_history"] = item_cate_history_batch_all
            res["mask"] = mask
            res["time"] = time_list_all
            res["time_diff"] = time_diff_batch
            res["time_from_first_action"] = time_from_first_action_batch
            res["time_to_now"] = time_to_now_batch
            return res

        else:
            instance_cnt = len(label_list)
            history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]
            max_seq_length_batch = self.max_seq_length
            item_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("int32")
            item_cate_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("int32")
            time_diff_batch = np.zeros((instance_cnt, max_seq_length_batch)).astype(
                "float32"
            )
            time_from_first_action_batch = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("float32")
            time_to_now_batch = np.zeros((instance_cnt, max_seq_length_batch)).astype(
                "float32"
            )
            mask = np.zeros((instance_cnt, max_seq_length_batch)).astype("float32")
            if self.hparams.use_cont_feat:
                cont_feat_batch = np.array(cont_feat_list).astype(
                    "float32"
                )
            else:
                cont_feat_batch = None

            for i in range(instance_cnt):
                this_length = min(history_lengths[i], max_seq_length_batch)
                item_history_batch_all[i, :this_length] = item_history_batch[i][
                    -this_length:
                ]
                item_cate_history_batch_all[i, :this_length] = item_cate_history_batch[
                    i
                ][-this_length:]
                mask[i, :this_length] = 1.0
                time_diff_batch[i, :this_length] = time_diff_list[i][-this_length:]
                time_from_first_action_batch[
                    i, :this_length
                ] = time_from_first_action_list[i][-this_length:]
                time_to_now_batch[i, :this_length] = time_to_now_list[i][-this_length:]

            res = {}
            res["labels"] = np.asarray(label_list, dtype=np.float32).reshape(-1, 1)
            res["users"] = np.asarray(user_list, dtype=np.float32)
            res["items"] = np.asarray(item_list, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list, dtype=np.int32)
            res["item_history"] = item_history_batch_all
            res["item_cate_history"] = item_cate_history_batch_all
            res["mask"] = mask
            res["time"] = np.asarray(time_list, dtype=np.float32)
            res["time_diff"] = time_diff_batch
            res["time_from_first_action"] = time_from_first_action_batch
            res["time_to_now"] = time_to_now_batch
            res["cont_feat"] = cont_feat_batch
            return res

    def gen_feed_dict(self, data_dict):
        """Construct a dictionary that maps graph elements to values.

        Args:
            data_dict (dict): A dictionary that maps string name to numpy arrays.

        Returns:
            dict: A dictionary that maps graph elements to numpy arrays.

        """
        if not data_dict:
            return dict()
        feed_dict = {
            self.labels: data_dict["labels"],
            self.users: data_dict["users"],
            self.items: data_dict["items"],
            self.cates: data_dict["cates"],
            self.item_history: data_dict["item_history"],
            self.item_cate_history: data_dict["item_cate_history"],
            self.mask: data_dict["mask"],
            self.time: data_dict["time"],
            self.time_diff: data_dict["time_diff"],
            self.time_from_first_action: data_dict["time_from_first_action"],
            self.time_to_now: data_dict["time_to_now"], 
        }
        if self.hparams.use_cont_feat:
            feed_dict[self.cont_feat] = data_dict["cont_feat"]
        return feed_dict
