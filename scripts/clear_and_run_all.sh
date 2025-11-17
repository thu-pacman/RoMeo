find reproduce/fig* \( -name "*.pdf" -o -name "*.log" \) -delete
find reproduce/tab* \( -name "*.pdf" -o -name "*.log" \) -delete

bash ./scripts/reproduce.sh tab1 | tee tab1.log
bash ./scripts/reproduce.sh tab2 | tee tab2.log
bash ./scripts/reproduce.sh fig7 | tee fig7.log
bash ./scripts/reproduce.sh fig8 | tee fig8.log
bash ./scripts/reproduce.sh fig9 | tee fig9.log
bash ./scripts/reproduce.sh fig10 | tee fig10.log
