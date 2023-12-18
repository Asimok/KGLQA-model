# nohup 运行 并echo 进称号
# 睡眠5小时（以秒为单位）
#sleep 18000
nohup python -u ablation_study_server.py > log.out 2>&1 &
echo $!


