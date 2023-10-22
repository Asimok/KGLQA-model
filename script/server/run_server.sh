# nohup 运行 并echo 进称号
nohup python -u single_chat_server.py > log.out 2>&1 &
echo $!


