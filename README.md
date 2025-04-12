Usage Instructions

To use the QoR prediction model:

Preprocess the circuit designs:

```bash
python main.py --mode preprocess --design_dir ./designs/ --processed_designs_path processed_designs.pt
```

Train the model:

```bash
python main.py --mode train --dataset_path dataset.csv --processed_designs_path processed_designs.pt --batch_size 16
```

Evaluate on test data:

```bash
python main.py --mode test --test_dataset_path test_dataset.csv --processed_designs_path processed_designs.pt
```

Make predictions:

```bash
python predict.py --mode batch --checkpoint_path checkpoints/best_model.pt --input_file predict_data.csv --output_file predictions.csv
```

Single design prediction:

```bash
python predict.py --mode single --checkpoint_path checkpoints/best_model.pt --design_name design1 --recipe "rewrite,refactor,balance"
```