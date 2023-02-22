#!/usr/bin/env python
# Copyright (C) 2023 - All rights reserved.
# Author Contact Info: Kourosh.Darvish@gmail.com
# SPDX-License-Identifier: BSD-3-Clause
#


import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    LSTM,
    Add,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    Layer,
    LeakyReLU,
    MaxPooling1D,
    Multiply,
    Reshape,
    Softmax,
)


class ReducedSum(Layer):
    def __init__(self, axis=None, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        print(f"[build] input_shape: {input_shape}")
        print(f"[build] input_shape[0]: {input_shape[0]}")
        return

    def call(self, inputs):
        output = tf.reduce_sum(inputs, axis=self.axis)
        return output

    def get_config(self):
        return {"axis": self.axis}


# not tested
class ProbabilisticSwitch(Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        print(f"[build] input_shape: {input_shape}")
        print(f"[build] input_shape[0]: {input_shape[0]}")
        return

    def call(self, experts, gate=None):
        arg_max = tf.math.argmax(gate, axis=-1)
        output = tf.fill(experts.shape[:-1], 0.0)
        for m in range(output.shape[0]):
            for t in range(output.shape[1]):
                output[m, t, :] = experts[m, t, :, arg_max[m, t]]
        return tf.convert_to_tensor(output)

    def get_config(self):
        return {"axis": self.axis}


def get_complex_gate_output(input_, number_categories, output_steps, reg_l1, reg_l2, dp_rate):
    output_ = Flatten(name="gate_nn")(input_)
    output_ = Dense(
        1024,
        activation="relu",
        kernel_regularizer=regularizers.l2(reg_l2),
        bias_regularizer=regularizers.l2(reg_l2),
    )(output_)
    output_ = BatchNormalization()(output_)
    output_ = Dropout(dp_rate)(output_)
    output_ = Dense(
        512,
        activation="relu",
        kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
        bias_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
    )(output_)
    output_ = BatchNormalization()(output_)
    output_ = Dropout(dp_rate)(output_)
    output_ = Dense(number_categories * output_steps)(output_)
    # ! default: axis=-1, normalization is done over last axis
    output_ = Reshape([output_steps, number_categories])(output_)
    output_ = Softmax()(output_)
    return output_


def get_simple_gate_output(input_, number_categories, output_steps, reg_l2, dp_rate):
    output_ = Dense(
        32,
        activation="relu",
        kernel_regularizer=regularizers.l2(reg_l2),
        bias_regularizer=regularizers.l2(reg_l2),
    )(input_)
    output_ = BatchNormalization()(output_)
    output_ = Dropout(dp_rate)(output_)
    output_ = Dense(
        number_categories * output_steps,
        kernel_regularizer=regularizers.l2(reg_l2),
        bias_regularizer=regularizers.l2(reg_l2),
    )(output_)
    output_ = BatchNormalization()(output_)
    output_ = Dropout(dp_rate)(output_)
    # ! default: axis=-1, normalization is done over last axis
    output_ = Reshape([output_steps, number_categories])(output_)
    output_ = Softmax()(output_)
    return output_


def get_dense_expert_output(input_, number_outputs, output_steps, reg_l2, dp_rate, expert_number):
    output_ = Flatten(name=f"expert{expert_number}_nn")(input_)
    output_ = Dense(
        output_steps * number_outputs,
        activation="relu",
        kernel_regularizer=regularizers.l2(reg_l2),
        bias_regularizer=regularizers.l2(reg_l2),
    )(output_)
    output_ = BatchNormalization()(output_)
    output_ = Dropout(dp_rate)(output_)
    output_ = Dense(
        output_steps * number_outputs,
        kernel_regularizer=regularizers.l2(reg_l2),
        bias_regularizer=regularizers.l2(reg_l2),
    )(output_)
    output_ = BatchNormalization()(output_)
    output_ = Dropout(dp_rate)(output_)
    output_ = Reshape([output_steps, number_outputs])(output_)
    return output_


def get_refined_lstm_expert_output(
    input_, number_experts_outputs, output_steps, reg_l1, reg_l2, dp_rate, expert_number
):
    h_expert = LSTM(
        78,
        name=f"expert_{expert_number}_nn",
        return_sequences=False,
        activation=LeakyReLU(),
        kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
    )(input_)

    h_expert = BatchNormalization()(h_expert)

    h_expert = Dense(output_steps * number_experts_outputs)(h_expert)

    h_expert = Reshape([output_steps, number_experts_outputs])(h_expert)

    return h_expert


def get_lstm_expert_output(input_, number_outputs, output_steps, reg_l2, dp_rate, expert_number):
    output_ = LSTM(
        64,
        name=f"expert{expert_number}_nn",
        return_sequences=True,
        activation="relu",
        kernel_regularizer=regularizers.l2(reg_l2),
        bias_regularizer=regularizers.l2(reg_l2),
        recurrent_regularizer=regularizers.l2(reg_l2),
        dropout=dp_rate,
        recurrent_dropout=dp_rate,
    )(input_)

    output_ = BatchNormalization()(output_)

    output_ = Flatten()(output_)

    output_ = Dense(
        output_steps * number_outputs,
        activation="relu",
        kernel_regularizer=regularizers.l2(reg_l2),
        bias_regularizer=regularizers.l2(reg_l2),
    )(output_)

    output_ = BatchNormalization()(output_)

    output_ = Dropout(dp_rate)(output_)

    output_ = Dense(
        output_steps * number_outputs,
        kernel_regularizer=regularizers.l2(reg_l2),
        bias_regularizer=regularizers.l2(reg_l2),
    )(output_)

    output_ = BatchNormalization()(output_)

    output_ = Dropout(dp_rate)(output_)

    output_ = Reshape([output_steps, number_outputs])(output_)

    return output_


def get_gate_selector_output_associative(
    h_gate,
    h_expert1,
    h_expert2,
    h_expert3,
    h_expert4,
    number_categories,
    number_experts_outputs,
    output_steps,
):
    h_expert1 = Reshape([output_steps, number_experts_outputs, 1])(h_expert1)
    h_expert2 = Reshape([output_steps, number_experts_outputs, 1])(h_expert2)
    h_expert3 = Reshape([output_steps, number_experts_outputs, 1])(h_expert3)
    h_expert4 = Reshape([output_steps, number_experts_outputs, 1])(h_expert4)
    experts = Concatenate(axis=-1)([h_expert1, h_expert2, h_expert3, h_expert4])
    print(f"experts shape: {experts.shape}")

    h_gate = Reshape([output_steps, 1, number_categories])(h_gate)
    print(f"h_gate shape: {h_gate.shape}")

    moe_output_ = Multiply()([experts, h_gate])
    print(f"moe_output_ shape: {moe_output_.shape}")
    moe_output = ReducedSum(name="moe_output", axis=3)(moe_output_)
    print(f"moe_output shape: {moe_output.shape}")

    return moe_output


# not tested
def get_gate_selector_output_competitive(
    h_gate,
    h_expert1,
    h_expert2,
    h_expert3,
    h_expert4,
    number_categories,
    number_experts_outputs,
    output_steps,
):
    experts = Concatenate(axis=-1)([h_expert1, h_expert2, h_expert3, h_expert4])
    moe_output = ProbabilisticSwitch(name="moe_output")(experts, h_gate)
    return moe_output


class GateLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.supports_masking = True  #?
        print(["__init__"])

    def build(self, input_shape):
        # purely used to check the layers size
        # the first input is always the gate outputs
        # all the rest are the experts outputs

        # gate output size: (, number of output time steps, number of categories)
        # the expert output size : (, number of output time steps, number of output features)
        # Consider:
        #   x = number of categories
        #   Then we should have x number of experts, so the size of the input tensor list should be x+1

        # to Enable later
        # if not isinstance(input_shape[0], tuple):
        #     raise ValueError('A gate layer should be called on a list of inputs.')
        if len(input_shape) < 2:
            raise ValueError(
                "A gate layer should be called "
                "on a list of at least 2 inputs. "
                "Got " + str(len(input_shape)) + " inputs."
            )
        batch_sizes = {s[0] for s in input_shape if s} - {None}
        print(f"[build] batch_sizes: {batch_sizes} , len(batch_sizes): {len(batch_sizes)}")
        if len(batch_sizes) > 1:
            raise ValueError(
                "Can not merge tensors with different " "batch sizes. Got tensors with shapes : " + str(input_shape)
            )

        # check if number of categories and input data are equal
        # print('[build] gate shape : {}, {}, {}'.format(input_shape[0].aslist(), input_shape[0][1][2]))
        gate_shape = input_shape[0]
        categories_size = gate_shape[-1]
        print(
            "[build] categories_size: {} ".format(
                categories_size,
            )
        )
        if categories_size + 1 != len(input_shape):
            raise ValueError(
                "Gate layer should have similar number of categories and input experts."
                "Got categories size of : "
                + str(categories_size)
                + " , Got number of experts: "
                + str(len(input_shape) - 1)
                + " Total number of input tensors: "
                + str(len(input_shape))
            )

        if input_shape[1] is None:
            output_shape = None
        else:
            output_shape = input_shape[1][1:]
        print(f"output_shape: {output_shape}")

    # Defines the computation from inputs to outputs
    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("A gate layer should be called on a list of inputs.")
        stacked_experts = tf.stack(inputs[1:], axis=-1)
        gate_shape = inputs[0].shape  # ! gate shape [No. samples, No of Time Steps (Ty), No. categories]
        num_output_features = inputs[1].shape[-1]
        # ! gate shape [No. samples, No of Time Steps (Ty), 1, No. categories]
        gate_output = inputs[0]
        reshaped_gate = tf.reshape(inputs[0], [gate_shape[0], gate_shape[1], 1, gate_shape[2]])
        new_gate_shape = reshaped_gate.shape
        broadcast_gate = tf.broadcast_to(
            reshaped_gate,
            shape=[gate_shape[0], gate_shape[1], num_output_features, gate_shape[2]],
        )
        # ! reduce sum about the categories, probability axis = 3 : the last one
        output = tf.reduce_sum(tf.multiply(broadcast_gate, stacked_experts), axis=3)
        return output


def get_complex_gate_output_ablation(input_, number_categories, output_steps, reg_l1, reg_l2, dp_rate):
    output_classification = LSTM(
        512,
        name="cls_lstm_layer_1",
        return_sequences=True,
        activation=LeakyReLU(),
        kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
    )(input_)
    output_classification = BatchNormalization()(output_classification)
    output_classification = LSTM(
        256,
        name="cls_lstm_layer_2",
        return_sequences=True,
        activation=LeakyReLU(),
        kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
    )(output_classification)

    output_classification = BatchNormalization()(output_classification)
    output_classification = LSTM(
        128,
        name="cls_lstm_layer_3",
        return_sequences=True,
        activation=LeakyReLU(),
        kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
    )(output_classification)
    output_classification = BatchNormalization()(output_classification)
    output_classification = LSTM(
        16,
        name="cls_lstm_layer_4",
        return_sequences=False,
        activation=LeakyReLU(),
        kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
    )(output_classification)
    output_classification = BatchNormalization()(output_classification)
    output_classification = Flatten()(output_classification)
    output_classification = Dense(number_categories * output_steps)(output_classification)
    # ! default: axis=-1, normalization is done over last axis
    output_classification = Reshape([output_steps, number_categories])(output_classification)
    output_classification = Softmax(name="gate_output_softmax")(output_classification)
    return output_classification


def get_refined_lstm_expert_output_ablation(
    input_, number_experts_outputs, output_steps, reg_l1, reg_l2, dp_rate, expert_number
):
    output_regression = LSTM(
        512,
        name=f"rgs_lstm_layer_1_expert{expert_number}",
        return_sequences=True,
        activation=LeakyReLU(),
        kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
    )(input_)
    output_regression = BatchNormalization()(output_regression)
    output_regression = LSTM(
        256,
        name=f"rgs_lstm_layer_2_expert{expert_number}",
        return_sequences=True,
        activation=LeakyReLU(),
        kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
    )(output_regression)
    output_regression = BatchNormalization()(output_regression)
    output_regression = LSTM(
        128,
        name=f"rgs_lstm_layer_3_expert{expert_number}",
        return_sequences=True,
        activation=LeakyReLU(),
        kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
    )(output_regression)
    output_regression = BatchNormalization()(output_regression)
    output_regression = LSTM(
        64,
        name=f"rgs_lstm_layer_4_expert{expert_number}",
        return_sequences=True,
        activation=LeakyReLU(),
        kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
    )(output_regression)
    output_regression = BatchNormalization()(output_regression)
    output_regression = Flatten()(output_regression)
    output_regression = Dense(output_steps * number_experts_outputs)(output_regression)
    output_regression = Reshape([output_steps, number_experts_outputs], name=f"moe_output_expert{expert_number}")(
        output_regression
    )
    return output_regression
