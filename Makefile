# MirageBench — yaygın iş akışları. `make help` ile listele.
.PHONY: help install install-train test lint synth-demo ingest run-dry report clean

PY ?= python

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

install: ## Çekirdek bağımlılıklar (CPU)
	$(PY) -m pip install -e ".[report,dev]"

install-train: ## Eğitim/çıkarım bağımlılıkları (GPU)
	$(PY) -m pip install -e ".[train]" && $(PY) -m pip install bpy

test: ## Testleri çalıştır (yavaş render testi hariç)
	$(PY) -m pytest tests/ -q --deselect tests/test_blender_smoke.py::test_render_one_bbox_bounds_drone

lint: ## ruff ile stil denetimi
	ruff check miragebench tests

synth-demo: ## 20 örnekli küçük sentetik demo üret (Blender/bpy gerekir)
	$(PY) -m miragebench.synth.cli --out data/synth/demo --n 20 --width 640 --height 480

ingest: ## Gerçek YOLO setini içeri al (DATA=/yol gerekir)
	$(PY) -m miragebench.data.cli --format yolo --root $(DATA) --out data/real --merge-classes

run-dry: ## Deney kollarının veri kompozisyonunu GPU'suz doğrula
	$(PY) -m miragebench.experiment.cli --dry-run \
	  --arm configs/arms/*.yaml --real-root data/real \
	  --synth full=data/synth/full --seeds 0

report: ## Sonuç JSON'larından rapor + grafik üret
	$(PY) -m miragebench.report.cli --results results --out docs/report

clean: ## Ara çıktıları temizle
	rm -rf work runs .pytest_cache .ruff_cache
