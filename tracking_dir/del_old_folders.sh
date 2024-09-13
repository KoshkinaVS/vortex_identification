#!/bin/bash

shopt -s extglob # Включаем поддержку расширенных глобальных выражений

for year in {1979..2018}; do
    base_dir="storage/kubrick/vkoshkina/data/LoRes_tracks/with_CS_speed/$year/tracks_AC"
    rm -rf "$base_dir"/+( * )/+( * ).csv # Удаляем все поддиректории tracks_С и их содержимое
done

