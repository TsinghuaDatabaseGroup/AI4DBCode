python train_model.py --version cols_4_distinct_1000_corr_5_skew_5 --num-gpus=1 --dataset=dmv --epochs=24 --warmups=8000 --bs=2048 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking
# modify modle name models/dmv-4.1MB-model16.077-data13.659-made-resmade-hidden256_256_256_256_256-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-colmask-20epochs-seed0.pt
python eval_model.py --testfilepath /home/jintao/naru/sql_truecard/ --version cols_4_distinct_1000_corr_5_skew_5 --table cols_4_distinct_1000_corr_5_skew_5 --alias cdcs --dataset=dmv --glob='<ckpt from above>' --num-queries=2000 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking

python train_model.py --version cols_4_distinct_1000_corr_5_skew_5 --num-gpus=1 --dataset=dmv --epochs=100 --warmups=12000 --bs=2048 --layers=0 --direct-io --column-masking --input-encoding=binary --output-encoding=one_hot
# modify modle name models/dmv-4.1MB-model16.077-data13.659-made-resmade-hidden256_256_256_256_256-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-colmask-20epochs-seed0.pt
python eval_model.py --testfilepath /home/jintao/naru/sql_truecard/ --version cols_4_distinct_1000_corr_5_skew_5 --table cols_4_distinct_1000_corr_5_skew_5 --alias cdcs --dataset=dmv --glob='<ckpt from above>' --num-queries=1000 --layers=0 --direct-io --column-masking --input-encoding=binary --output-encoding=one_hot
