# nohup 运行 并echo 进称号
nohup python -u captions_server.py > captions.log 2>&1 &
echo $!


