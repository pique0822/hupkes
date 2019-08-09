SINGULARITY_IMG=/om2/user/jgauthie/singularity_images/deepo-cpu.simg
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd -P )"


for k in $(seq 1 20)
do
	echo Training Model $k

	singularity exec -B /om/group/cpl -B "$PROJECT_PATH" "$SINGULARITY_IMG" python3 train_GRU.py \
		--model_save "models/hupkes_model_${k}.mdl" \
		--num_epochs 500 >> "models/training_model_${k}.txt"
done
