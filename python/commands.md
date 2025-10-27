python asr2eaf.py \
    --wav /Users/sjhuskey/JE_07092025_Hapong_Aniep.wav \
    --pause-based \
    --min-pause-ms 150 \
    --rel-thresh-db 16 \
    --keep-silence-ms 20 \
    --out-csv /Users/sjhuskey/enenlhet-output/csv/JE_07092025_Hapong_Aniep.csv \
    --model sjhuskey/enenlhet-wav2vec2-model \
    --segment-dir /Users/sjhuskey/audio-segments \
    --to-eaf --new-eaf /Users/sjhuskey/enenlhet-output/eaf/JE_07092025_Hapong_Aniep.eaf \
    --tier-name ASR

python asr2eaf.py \
  --batch-dir /Users/sjhuskey/enenlhet-raw-data \
  --recursive \
  --pause-based \
  --min-pause-ms 150 \
  --rel-thresh-db 16 \
  --keep-silence-ms 20 \
  --model sjhuskey/enenlhet-wav2vec2-model \
  --out-csv-template /Users/sjhuskey/enenlhet-output/csv/{stem}.csv \
  --to-eaf \
  --new-eaf-template /Users/sjhuskey/enenlhet-output/eaf/{stem}.eaf \
  --tier-name ASR_pause
