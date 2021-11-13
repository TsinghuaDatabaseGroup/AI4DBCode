import os

os.chdir('/home/jintao/naru/')
# cols_4_distinct_1000_corr_5_skew_5
'''
for version in ['cols_2_distinct_1000_corr_6_skew_6', 'cols_4_distinct_1000_corr_6_skew_6', 'cols_6_distinct_1000_corr_6_skew_6', 
                'cols_8_distinct_1000_corr_6_skew_6', 'cols_4_distinct_10_corr_6_skew_6', 'cols_4_distinct_100_corr_6_skew_6', 
                'cols_4_distinct_10000_corr_6_skew_6', 'cols_4_distinct_1000_corr_2_skew_6', 'cols_4_distinct_1000_corr_4_skew_6', 
                'cols_4_distinct_1000_corr_8_skew_6', 'cols_4_distinct_1000_corr_6_skew_2', 'cols_4_distinct_1000_corr_6_skew_4', 
                'cols_4_distinct_1000_corr_6_skew_8']:
'''
for version in ['cols_4_distinct_10_corr_6_skew_6']:
    # version = 'cols_' + str(cols) + '_distinct_' + str(distinct) + '_corr_' + str(corr) + '_skew_' + str(skew)

    # train
    os.system(
        'python train_model.py --version ' + version + ' --num-gpus=1 --dataset=dmv --epochs=60 --warmups=8000 --bs=2048 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking')
    # python train_model.py --version cols_4_distinct_1000_corr_5_skew_5 --num-gpus=1 --dataset=dmv --epochs=100 --warmups=12000 \
    # --bs=2048 --layers=0 --direct-io --column-masking --input-encoding=binary --output-encoding=one_hot

    # test
    os.system(
        'python eval_model.py --testfilepath /home/jintao/naru/sql_truecard/ --version ' + version + ' --table ' + version + ' --alias cdcs --dataset=dmv --glob=\'<ckpt from above>\' --num-queries=1000 --residual --layers=5 --fc-hiddens=256 --direct-io --column-masking')
    # python eval_model.py --testfilepath /home/jintao/naru/sql_truecard/ --version cols_4_distinct_1000_corr_5_skew_5 \
    # --table cols_4_distinct_1000_corr_5_skew_5 --alias cdcs --dataset=dmv --glob='<ckpt from above>' --num-queries=1000 --layers=0 \
    # --direct-io --column-masking --input-encoding=binary --output-encoding=one_hot

    print(version + 'is OK.')
