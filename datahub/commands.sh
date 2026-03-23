zip -r IXI_2D.zip ./IXI_2D -x "*__MACOSX*" -x "*._*" -x "*.DS_Store"
scp ./data/raw/IXI_2D.zip kshahi@dsmlp-login.ucsd.edu:~/data/
rm -rf ~/data/IXI_2D_synth_trip/
du -sh ~
unzip ~/data/IXI_2D_synth_trip.zip -x "__MACOSX/*" -d ~/data/IXI_2D_synth_trip/
find ~/data/IXI_2D_synth_trip/ -name "._*" -delete
tmux new -s myjob
tmux attach -t myjob
python -m zipfile -c runs.zip runs