#!/bin/bash
set -e

NUM_JOB=${NUM_JOB:-36}
echo "| Training MFA using ${NUM_JOB} cores."
BASE_DIR=data/processed/$CORPUS
MODEL_NAME=${MODEL_NAME:-"mfa_model"}
PRETRAIN_MODEL_NAME=${PRETRAIN_MODEL_NAME:-"mfa_model_pretrain"}
MFA_INPUTS=${MFA_INPUTS:-"mfa_inputs"}
MFA_OUTPUTS=${MFA_OUTPUTS:-"mfa_outputs"}
MFA_CMD=${MFA_CMD:-"train"}
rm -rf $BASE_DIR/mfa_outputs_tmp
if [ "$MFA_CMD" = "train" ]; then
  mfa train $BASE_DIR/$MFA_INPUTS $BASE_DIR/mfa_dict.txt $BASE_DIR/mfa_outputs_tmp -t $BASE_DIR/mfa_tmp -o $BASE_DIR/$MODEL_NAME.zip --clean -j $NUM_JOB --config_path mfa_usr/mfa_train_config.yaml
elif [ "$MFA_CMD" = "adapt" ]; then
  python mfa_usr/mfa.py adapt \
  $BASE_DIR/$MFA_INPUTS \
  $BASE_DIR/mfa_dict.txt \
  $BASE_DIR/$PRETRAIN_MODEL_NAME.zip \
  $BASE_DIR/$MODEL_NAME.zip \
  $BASE_DIR/mfa_outputs_tmp \
  -t $BASE_DIR/mfa_tmp --clean -j $NUM_JOB
fi
rm -rf $BASE_DIR/mfa_tmp $BASE_DIR/$MFA_OUTPUTS
mkdir -p $BASE_DIR/$MFA_OUTPUTS
find $BASE_DIR/mfa_outputs_tmp -regex ".*\.TextGrid" -print0 | xargs -0 -i mv {} $BASE_DIR/$MFA_OUTPUTS/
if [ -e "$BASE_DIR/mfa_outputs_tmp/unaligned.txt" ]; then
  cp $BASE_DIR/mfa_outputs_tmp/unaligned.txt $BASE_DIR/
fi
rm -rf $BASE_DIR/mfa_outputs_tmp