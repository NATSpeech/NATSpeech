import utils.commons.single_thread_env  # NOQA
import os
import subprocess
from utils.commons.hparams import hparams, set_hparams


def adapt_mfa_align():
    CORPUS = hparams['processed_data_dir'].split("/")[-1]
    print(f"| Run MFA for {CORPUS}.")
    NUM_JOB = int(os.getenv('N_PROC', os.cpu_count()))
    subprocess.check_call(
        f'CORPUS={CORPUS} NUM_JOB={NUM_JOB} bash scripts/run_mfa_adapt.sh',
        shell=True)


if __name__ == '__main__':
    set_hparams(print_hparams=False)
    adapt_mfa_align()
