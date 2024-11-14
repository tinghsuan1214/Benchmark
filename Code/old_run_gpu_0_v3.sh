for i in {1..1} # Here you can set how many to train at a time
do
    directory_path="/mnt/4TB/tinghsuan/Benchmark/Code/training/model_$i"
    log_file_name="model_$i"
    model_type="LSTM_nodes3"
    type="[You can name it yourself]"
    data_size=100
    window_size=48
    batch_size=256
    epochs=100
    patience=20
    user_index=0
    num_linear_layer=1
    c_in=1
    c_out=64
    num_cells=1
    root_dir="/mnt/4TB/tinghsuan/Benchmark/Code/training/All_model_info"
    arch_str_file_path='/mnt/4TB/tinghsuan/Benchmark/Code/training/All_model_info/All_arch_str.txt'

    python old_main_cuda_0_v3.py "$directory_path" "$log_file_name" "$model_type" "$type" "$data_size" "$window_size" "$batch_size" "$epochs" "$patience" "$user_index" "$num_linear_layer" "$c_in" "$c_out" "$num_cells" "$root_dir" "$arch_str_file_path"
    python all_arch_str.py "$root_dir" "$arch_str_file_path"
done

