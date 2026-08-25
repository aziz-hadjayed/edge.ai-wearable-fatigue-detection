/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-08-23T18:39:17+0100
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[3]) {
   { .size_bytes = 540, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 135}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "float_input_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S32, .shape = {1, (const int32_t[1]){1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "label_output0" },
   { .size_bytes = 16, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "label_output1" }
  },
  .nodes = (const stai_node_details[1]){
    {.id = 1, .type = AI_LAYER_TREE_ENSEMBLE_CLASSIFIER_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {2, (const int32_t[2]){1, 2}} } /* label */
  },
  .n_nodes = 1
};
#endif

