#!/bin/bash
#
# SPDX-FileCopyrightText: 2025 SAP SE or an SAP affiliate company
#
# SPDX-License-Identifier: Apache-2.0
#
# Script to download and prepare ToxiGen dataset for evaluation
# Usage: bash scripts/prepare_data.sh

# Exit on error
set -e

echo "Downloading ToxiGen dataset..."

# Create directory
mkdir -p data/eval/toxigen

# Download hate speech prompts
for minority_group in asian black chinese jewish latino lgbtq mental_disability mexican middle_east muslim native_american physical_disability women
do
    echo "Downloading hate prompts for ${minority_group}..."
    wget -q -O data/eval/toxigen/hate_${minority_group}.txt https://raw.githubusercontent.com/microsoft/TOXIGEN/main/prompts/hate_${minority_group}_1k.txt
done

# Download neutral prompts
for minority_group in asian black chinese jewish latino lgbtq mental_disability mexican middle_east muslim native_american physical_disability women
do
    echo "Downloading neutral prompts for ${minority_group}..."
    wget -q -O data/eval/toxigen/neutral_${minority_group}.txt https://raw.githubusercontent.com/microsoft/TOXIGEN/main/prompts/neutral_${minority_group}_1k.txt
done

echo "Data preparation complete. Files saved to data/eval/toxigen/"
