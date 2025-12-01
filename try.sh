



accelerate launch --multi_gpu --num_processes N --num_machines 1 --mixed_precision bf16 train_t2i_discrete.py \
            --config=configs/t2i_training_demo.py



accelerate launch --num_processes 1 --num_machines 1 --num_cpu_threads_per_process 4 --mixed_precision bf16 \
        train_t2i_edit.py \
        --config=configs/t2i_training_anyedit.py


accelerate launch --num_processes 8 --num_machines 1 --num_cpu_threads_per_process 4 --mixed_precision bf16 \
        train_t2i_edit.py \
        --config=configs/t2i_training_anyedit.py




accelerate launch --num_processes 8 --num_machines 1 --num_cpu_threads_per_process 4 --mixed_precision bf16 \
        train_t2i_edit.py \
        --config=configs/t2i_training_anyedit_edit.py


accelerate launch --num_processes 8 --num_machines 1 --num_cpu_threads_per_process 4 --mixed_precision bf16 \
        train_t2i_edit.py \
        --config=configs/t2i_training_anyedit_edit_x.py


accelerate launch --num_processes 8 --num_machines 1 --num_cpu_threads_per_process 4 --mixed_precision bf16 \
        train_t2i_edit.py \
        --config=configs/t2i_training_anyedit_edit_s.py




accelerate launch --num_processes 8 --num_machines 1 --num_cpu_threads_per_process 4 --mixed_precision bf16 \
        train_t2i_edit.py \
        --config=configs/t2i_training_anyedit_edit_direct.py


accelerate launch --num_processes 8 --num_machines 1 --num_cpu_threads_per_process 4 --mixed_precision bf16 \
        train_t2i_edit.py \
        --config=configs/t2i_training_anyedit_edit_x_direct.py


accelerate launch --num_processes 8 --num_machines 1 --num_cpu_threads_per_process 4 --mixed_precision no \
        train_t2i_edit.py \
        --config=configs/t2i_training_anyedit_edit_x_direct.py


accelerate launch --num_processes 8 --num_machines 1 --num_cpu_threads_per_process 4 --mixed_precision no \
        train_t2i_edit.py \
        --config=configs/t2i_training_anyedit_edit_x_direct_naive.py


accelerate launch --num_processes 8 --num_machines 1 --num_cpu_threads_per_process 4 --mixed_precision bf16 \
        train_t2i_edit.py \
        --config=configs/t2i_training_anyedit_edit_x_direct_naive.py


accelerate launch --num_processes 8 --num_machines 1 --num_cpu_threads_per_process 4 --mixed_precision bf16 \
        train_t2i_edit.py \
        --config=configs/t2i_training_anyedit_edit_s_direct.py




accelerate launch --num_processes 8 --num_machines 1 --num_cpu_threads_per_process 4 --mixed_precision bf16 \
        train_t2i_edit.py \
        --config=configs/t2i_training_anyedit_edit_direct_pixart.py