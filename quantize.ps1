# quantize.ps1
$ErrorActionPreference = "Stop"

# Пути к файлам
$INPUT_ADAPTER = "P:\AI_Lenin\models\saiga\lenin_model.f16.gguf"
$OUTPUT_ADAPTER = "P:\AI_Lenin\models\saiga\lenin_model.q4_k.gguf"
$QUANT_TYPE = "q4_k"  # q4_0, q4_1, q5_0, q5_1, q8_0

# Команда квантования
python -m llama_cpp.server --model $INPUT_ADAPTER --quantize $QUANT_TYPE --output $OUTPUT_ADAPTER

# Проверка результата
if (Test-Path $OUTPUT_ADAPTER) {
    $size = (Get-Item $OUTPUT_ADAPTER).Length / 1GB
    Write-Host "✅ Адаптер успешно квантован! Размер: $($size.ToString('0.00')) GB"
} else {
    Write-Host "❌ Ошибка квантования!"
}