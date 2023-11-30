#!/bin/bash
chr=$1
./admixture data/ASW_CEU_YRI_${chr}.bed -j10 --supervised 2 -B10000
