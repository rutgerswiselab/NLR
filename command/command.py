# NLR
'python main.py --rank 0 --model_name NLR --optimizer Adam --lr 0.001 --dataset logic1k_3k-15-15 --metric accuracy,rmse --l2 1e-05 --r_length 0.001 --r_logic 0.01 --random_seed 2018 --gpu 0'
'python main.py --rank 0 --model_name NLR --optimizer Adam --lr 0.001 --dataset logic10k_30k-15-15 --metric accuracy,rmse --l2 0.001 --r_length 0.001 --r_logic 0.1 --random_seed 2018 --gpu 0'

# NLR for Recommendation
'python main.py --rank 1 --model_name NLRRec --optimizer Adam --lr 0.001 --dataset ml100k01-1-5 --metric ndcg@10,precision@1 --max_his 10 --sparse_his 0 --neg_his 1 --l2 1e-4 --r_logic 1e-06 --r_length 1e-4 --random_seed 2018 --gpu 0'
'python main.py --rank 1 --model_name NLRRec --optimizer Adam --lr 0.001 --dataset 5Electronics01-1-5 --metric ndcg@10,precision@1 --max_his 10 --sparse_his 0 --neg_his 1 --l2 1e-4 --r_logic 1e-05 --r_length 1e-6 --random_seed 2018 --gpu 0'

# logic1k_3k-15-15
'python main.py --rank 0 --model_name RNNLogic --optimizer Adam --lr 0.001 --dataset logic1k_3k-15-15 --metric accuracy,rmse --rnn_bi 1 --rnn_type rnn --l2 1e-06 --random_seed 2018 --gpu 0'
'python main.py --rank 0 --model_name RNNLogic --optimizer Adam --lr 0.001 --dataset logic1k_3k-15-15 --metric accuracy,rmse --rnn_bi 1 --rnn_type lstm --l2 0.01 --random_seed 2018 --gpu 0'
'python main.py --rank 0 --model_name CNNLogic --optimizer Adam --lr 0.001 --dataset logic1k_3k-15-15 --metric accuracy,rmse --l2 1e-06 --random_seed 2018 --gpu 0'

# logic10k_30k-15-15
'python main.py --rank 0 --model_name RNNLogic --optimizer Adam --lr 0.001 --dataset logic10k_30k-15-15 --metric accuracy,rmse --rnn_bi 1 --rnn_type rnn --l2 0.001 --random_seed 2018 --gpu 0'
'python main.py --rank 0 --model_name RNNLogic --optimizer Adam --lr 0.001 --dataset logic10k_30k-15-15 --metric accuracy,rmse --rnn_bi 1 --rnn_type lstm --l2 1e-05 --random_seed 2018 --gpu 0'
'python main.py --rank 0 --model_name CNNLogic --optimizer Adam --lr 0.001 --dataset logic10k_30k-15-15 --metric accuracy,rmse --l2 0.01 --random_seed 2018 --gpu 0'

# ml100k01-1-5
'python main.py --rank 1 --model_name NAIS -—optimizer Adam --lr 0.001 --dataset ml100k01-1-5 --metric ndcg@10,precision@1 --l2 1e-5 --all_his 1 --eval_batch_size 512 --random_seed 2018 --gpu 0'
'python main.py --rank 1 --model_name STAMP --optimizer Adam --lr 0.001 --dataset ml100k01-1-5 --metric ndcg@10,precision@1 --l2 1e-6 --max_his 10 --sparse_his 0 --neg_his 0 --neg_emb 0 --random_seed 2018 --gpu 0'
'python main.py --rank 1 --model_name GRU4Rec —optimizer Adam --lr 0.001 --dataset ml100k01-1-5 --metric ndcg@10,precision@1 --l2 1e-5 --max_his 10 --sparse_his 0 --neg_his 0 --neg_emb 0 --random_seed 2018 --gpu 0'
'python main.py --rank 1 --model_name NARM —optimizer Adam --lr 0.001 --dataset ml100k01-1-5 --metric ndcg@10,precision@1 --l2 1e-5 --max_his 10 --sparse_his 0 --neg_his 0 --neg_emb 0 --random_seed 2018 --gpu 0'

# 5Electronics01-1-5
'python main.py --rank 1 --model_name NAIS -—optimizer Adam --lr 0.001 --dataset 5Electronics01-1-5 --metric ndcg@10,precision@1 --l2 1e-3 --all_his 1 --eval_batch_size 32 --random_seed 2018 --gpu 0'
'python main.py --rank 1 --model_name STAMP --optimizer Adam --lr 0.001 --dataset 5Electronics01-1-5 --metric ndcg@10,precision@1 --l2 1e-4 --max_his 10 --sparse_his 0 --neg_his 0 --neg_emb 0 --random_seed 2018 --gpu 0'
'python main.py --rank 1 --model_name GRU4Rec --optimizer Adam --lr 0.001 --dataset 5Electronics01-1-5 --metric ndcg@10,precision@1 --l2 1e-4 --max_his 10 --sparse_his 0 --neg_his 0 --neg_emb 0 --random_seed 2018 --gpu 0'
'python main.py --rank 1 --model_name NARM --optimizer Adam --lr 0.001 --dataset 5Electronics01-1-5 --metric ndcg@10,precision@1 --l2 1e-4 --max_his 10 --sparse_his 0 --neg_his 0 --neg_emb 0 --random_seed 2018 --gpu 0'
