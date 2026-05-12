python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 96}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 256, "hidden_size": 64, "horizon": 96, "label_len": 48, "lr": 0.02, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "ETTh1/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 192}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 256, "hidden_size": 512, "horizon": 192, "label_len": 48, "lr": 0.02, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "ETTh1/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 336}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 256, "hidden_size": 512, "horizon": 336, "label_len": 48, "lr": 0.03, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "ETTh1/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 720}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 256, "hidden_size": 512, "horizon": 720, "label_len": 48, "lr": 0.03, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "ETTh1/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon": 96}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 256, "hidden_size": 64, "horizon": 96, "label_len": 48, "lr": 0.02, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "ETTh2/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon": 192}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 256, "hidden_size": 512, "horizon": 192, "label_len": 48, "lr": 0.02, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "ETTh2/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon": 336}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 256, "hidden_size": 512, "horizon": 336, "label_len": 48, "lr": 0.03, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "ETTh2/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon": 720}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 256, "hidden_size": 512, "horizon": 720, "label_len": 48, "lr": 0.03, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "ETTh2/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm1.csv" --strategy-args '{"horizon": 96}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 128, "horizon": 96, "label_len": 48, "lr": 0.02, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "ETTm1/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm1.csv" --strategy-args '{"horizon": 192}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 128, "horizon": 192, "label_len": 48, "lr": 0.02, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "ETTm1/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm1.csv" --strategy-args '{"horizon": 336}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 128, "horizon": 336, "label_len": 48, "lr": 0.02, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "ETTm1/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm1.csv" --strategy-args '{"horizon": 720}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 256, "hidden_size": 128, "horizon": 720, "label_len": 48, "lr": 0.005, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "ETTm1/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon": 96}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 128, "horizon": 96, "label_len": 48, "lr": 0.02, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "ETTm2/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon": 192}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 128, "horizon": 192, "label_len": 48, "lr": 0.02, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "ETTm2/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon": 336}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 128, "horizon": 336, "label_len": 48, "lr": 0.02, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "ETTm2/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon": 720}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 256, "hidden_size": 128, "horizon": 720, "label_len": 48, "lr": 0.005, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "ETTm2/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon": 96}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 1, "batch_size": 16, "hidden_size": 512, "horizon": 96, "label_len": 48, "lr": 0.005, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Solar/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon": 192}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 1, "batch_size": 16, "hidden_size": 512, "horizon": 192, "label_len": 48, "lr": 0.002, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Solar/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon": 336}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 1, "batch_size": 16, "hidden_size": 1024, "horizon": 336, "label_len": 48, "lr": 0.0005, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Solar/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon": 720}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 1, "batch_size": 16, "hidden_size": 1024, "horizon": 720, "label_len": 48, "lr": 0.0005, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Solar/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon": 96}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 128, "horizon": 96, "label_len": 48, "lr": 0.02, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Weather/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon": 192}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 128, "horizon": 192, "label_len": 48, "lr": 0.02, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Weather/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon": 336}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 128, "horizon": 336, "label_len": 48, "lr": 0.02, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Weather/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon": 720}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 256, "hidden_size": 128, "horizon": 720, "label_len": 48, "lr": 0.005, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Weather/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 96}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 1, "batch_size": 16, "hidden_size": 512, "horizon": 96, "label_len": 48, "lr": 0.005, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Electricity/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 192}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 1, "batch_size": 16, "hidden_size": 512, "horizon": 192, "label_len": 48, "lr": 0.002, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Electricity/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 336}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 1, "batch_size": 16, "hidden_size": 1024, "horizon": 336, "label_len": 48, "lr": 0.0005, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Electricity/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 720}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 1, "batch_size": 16, "hidden_size": 1024, "horizon": 720, "label_len": 48, "lr": 0.0005, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Electricity/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 96}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 1, "batch_size": 16, "hidden_size": 512, "horizon": 96, "label_len": 48, "lr": 0.005, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Traffic/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 192}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 1, "batch_size": 16, "hidden_size": 512, "horizon": 192, "label_len": 48, "lr": 0.002, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Traffic/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 336}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 1, "batch_size": 16, "hidden_size": 1024, "horizon": 336, "label_len": 48, "lr": 0.0005, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Traffic/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 720}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 1, "batch_size": 16, "hidden_size": 1024, "horizon": 720, "label_len": 48, "lr": 0.0005, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Traffic/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 96}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 1, "batch_size": 16, "hidden_size": 512, "horizon": 96, "label_len": 48, "lr": 0.002, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Traffic/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 96}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 1, "batch_size": 16, "hidden_size": 1024, "horizon": 96, "label_len": 48, "lr": 0.0005, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Traffic/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon": 96}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 256, "hidden_size": 64, "horizon": 96, "label_len": 48, "lr": 0.02, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Exchange/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon": 192}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 256, "hidden_size": 512, "horizon": 192, "label_len": 48, "lr": 0.02, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Exchange/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon": 336}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 256, "hidden_size": 512, "horizon": 336, "label_len": 48, "lr": 0.03, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Exchange/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon": 720}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 256, "hidden_size": 512, "horizon": 720, "label_len": 48, "lr": 0.03, "norm": true, "seq_len": 96}' --gpus 2 --num-workers 1 --timeout 60000 --save-path "Exchange/Amplifier"

