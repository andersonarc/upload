#!/bin/bash
split -b 50M "$1" "${1}.part"