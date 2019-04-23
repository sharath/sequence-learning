for ((i = 0; i <= 30; i++))
do
    sbatch -p 1080ti-long --gres=gpu:1 run.sh $i
    sbatch -p 1080ti-long --gres=gpu:1 run.sh $i
done
