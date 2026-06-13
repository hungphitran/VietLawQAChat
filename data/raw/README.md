# Extract dataset

## Requirement: 7zip

- macOS: `brew install p7zip`
- Linux: `sudo apt install p7zip-full`
- Windows: https://www.7-zip.org/

## macOS / Linux:

```bash
cat dataset_part_* > dataset.7z
7z x dataset.7z
```

## Windows:

```cmd
copy /b dataset_part_aa + dataset_part_ab + dataset_part_ac + dataset_part_ad + dataset_part_ae dataset.7z
7z x dataset.7z
```
