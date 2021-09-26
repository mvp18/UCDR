# DomainNet
CUDA_VISIBLE_DEVICES=0 python3 main.py -hd sketch -sd quickdraw -gd real -aux 1 -wcce 1 -wmse 1 -alpha 1 -wrat 1 -beta 1 -bs 60 -mixl img -es 15
CUDA_VISIBLE_DEVICES=0 python3 main.py -hd quickdraw -sd sketch -gd real -aux 1 -wcce 1 -wmse 1 -alpha 1 -wrat 1 -beta 2 -bs 60 -mixl img -es 15
CUDA_VISIBLE_DEVICES=0 python3 main.py -hd painting -sd infograph -gd real -aux 1 -wcce 1 -wmse 1 -alpha 1 -wrat 1 -beta 1 -bs 60 -mixl img -es 15
CUDA_VISIBLE_DEVICES=0 python3 main.py -hd infograph -sd painting -gd real -aux 1 -wcce 1 -wmse 1 -alpha 1 -wrat 1 -beta 1 -bs 60 -mixl img -es 15
CUDA_VISIBLE_DEVICES=0 python3 main.py -hd clipart -sd painting -gd real -aux 1 -wcce 1 -wmse 1 -alpha 1 -wrat 1 -beta 1 -bs 60 -mixl img -es 15

# Sketchy
CUDA_VISIBLE_DEVICES=0 python3 main.py -wcce 1 -wmse 1 -alpha 2 -wrat 0.5 -data Sketchy -bs 64 -mixl img -es 15 -eccv 1
CUDA_VISIBLE_DEVICES=0 python3 main.py -wcce 1 -wmse 1 -alpha 1 -wrat 1 -data Sketchy -bs 64 -mixl img -es 15 -eccv 0

# TUBerlin
CUDA_VISIBLE_DEVICES=0 python3 main.py -wcce 1 -wmse 1 -alpha 1 -wrat 1 -data TUBerlin -bs 64 -mixl img -es 15