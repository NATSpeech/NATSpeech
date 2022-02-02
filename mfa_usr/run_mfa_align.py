import argparse
import glob
import importlib
import os
import subprocess
from utils.commons.hparams import set_hparams, hparams
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.os_utils import remove_file
from utils.text.encoding import get_encoding


def process_item(idx, txt_fn):
    base_fn = os.path.splitext(txt_fn)[0]
    basename = os.path.basename(base_fn)
    if os.path.exists(base_fn + '.wav'):
        wav_fn = base_fn + '.wav'
    elif os.path.exists(base_fn + '.mp3'):
        wav_fn = base_fn + '.mp3'
    else:
        return
    # process text
    encoding = get_encoding(txt_fn)
    with open(txt_fn, encoding=encoding) as f:
        txt_raw = " ".join(f.readlines()).strip()
    phs, _, phs_for_align, _ = preprocesser.process_text(txt_processor, txt_raw, hparams['preprocess_args'])
    os.makedirs(f'{mfa_process_dir}/{basename}', exist_ok=True)
    with open(f'{mfa_process_dir}/{basename}/{basename}.lab', 'w') as f:
        f.write(phs_for_align)
    # process wav
    new_wav_fn = preprocesser.process_wav(basename, wav_fn, mfa_process_dir, preprocess_args)
    subprocess.check_call(f'cp "{new_wav_fn}" "{mfa_process_dir}/{basename}/{basename}.wav"', shell=True)


if __name__ == "__main__":
    set_hparams()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_dir', type=str, default='', help='input dir')
    args, unknown = parser.parse_known_args()
    input_dir = args.input_dir
    processed_data_dir = hparams['processed_data_dir']
    preprocess_args = hparams['preprocess_args']
    preprocess_args['sox_to_wav'] = True
    preprocess_args['trim_all_sil'] = True
    # preprocess_args['trim_sil'] = True
    # preprocess_args['denoise'] = True

    pkg = ".".join(hparams["preprocess_cls"].split(".")[:-1])
    cls_name = hparams["preprocess_cls"].split(".")[-1]
    process_cls = getattr(importlib.import_module(pkg), cls_name)
    preprocesser = process_cls()
    txt_processor = preprocesser.txt_processor
    num_workers = int(os.getenv('N_PROC', os.cpu_count()))

    mfa_process_dir = f'{input_dir}/mfa_inputs'
    remove_file(mfa_process_dir, f'{input_dir}/mfa_tmp')
    os.makedirs(mfa_process_dir, exist_ok=True)
    os.makedirs(f'{mfa_process_dir}/processed_tmp', exist_ok=True)
    for res in multiprocess_run_tqdm(
            process_item, list(enumerate(glob.glob(f'{input_dir}/*.txt')))):
        pass
    remove_file(f'{mfa_process_dir}/processed_tmp')
    subprocess.check_call(
        f'mfa align {mfa_process_dir} '  # process dir
        f'{hparams["processed_data_dir"]}/mfa_dict.txt '  # dict
        f'{input_dir}/mfa_model.zip '  # model
        f'{input_dir}/mfa_outputs -t {input_dir}/mfa_tmp  -j {num_workers} '
        f' && cp -rf {input_dir}/mfa_outputs/*/* {input_dir}/'
        f' && cp -rf {mfa_process_dir}/*/* {input_dir}/'
        f' && rm -rf {input_dir}/mfa_tmp {input_dir}/mfa_outputs {mfa_process_dir}',  # remove tmp dir
        shell=True)
