# LL7 - PPO porównanie

Porównaj działanie różnych implementacji algorytmu PPO. Wybierz co najmniej 2 różne wersje z innych bibliotek (jedna z wersji może być implementacją własną) i przetestuj ich działanie na środowisku [HalfCheetah-v2](https://gym.openai.com/envs/HalfCheetah-v2/).

## Struktura repozytorium
Repozytorium zostało podzielone na następujące sekcje:
* folder główny - pliki instalacyjne i dokumentacyjne
* data - rezultaty eksperymentów tj. pliki z przebiegów uczenia i modele
* plots - wykresy wygenerowane w notatniku Jupyter w celach prezentacji
* src - kod źródłowy zawierający pliki do trenowania, ewaluacji i prezentacji wyników

## Instalacja środowiska
W celu instalacji środowiska należy uruchomić skrypt `setup.sh`:
```sh
source setup.sh
```
Skrypt ten:
* stworzy wirtualne środowisko o nazwie `usd-ppo` i je aktywuje
* zainstaluje `spinningup` z repozytorium i je zainstaluje
* zainstaluje `mujoco`
* zainstaluje inne wymagane biblioteki z pliku `requirements.txt`
## Trening
Skrypty do uruchamiania treningu modeli znajdują się folderze `src` i prezentują się następująco:
* `train_spinup_ppo_tf.py` - trening PPO w Spinning Up - Tensorflow
* `train_spinup_ppo_torch.py` - trening PPO w Spinning Up - PyTorch
* `train_spinup_rllib_ray.py` - trening PPO w RLlib

Wszystkie skrypty wykonują *grid search* na parametrach opisanych w dokumentacji tj:
* `gamma` [0.99, 0.94, 0.89]
* `clip_ratio` [0.1, 0.2, 0.3]
* `target_kl` [0.01, 0.05]

Przykład wywołania:
```sh
python train_spinup_ppo_tf.py
```
## Ewaluacja
Do ewaluacji wytrenowanych agentów ze spinnup-a używamy wbudowanego modułu [`run`](https://spinningup.openai.com/en/latest/user/saving_and_loading.html#if-environment-saves-successfully), który dostarcza taką funkcjonalność. Przykładowe wywołanie wygląda następująco:
```sh
python -m spinup.run test_policy data/ppo-spinup-tf1-half-cheetah_halfcheetah-v2_gam0-89_cli0-1_tar0-01/ppo-spinup-tf1-half-cheetah_halfcheetah-v2_gam0-89_cli0-1_tar0-01_s50
```


W przypadku RLlib przygotowany do tego celu został skrypt `eval_ppo_rllib_ray.py`, który to przyjmuje następujące argumenty:
* path_to_checkpoint - ścieżka do *checkpoint-u*
* path_to_config - ścieżka do pliku z konfiguracją

```sh
python eval_ppo_rllib_ray.py --path_to_config models/rllib_tanh/ppo-rllib-half-cheetah-tanh-gamma0_89-clip_ratio0_1-target_kl0_01-v0.json --path_to_checkpoint models/rllib_tanh/ppo-rllib-half-cheetah-tanh-gamma0_89-clip_ratio0_1-target_kl0_01-v0/checkpoint_001000/checkpoint-1000
```
## Prezentacja rezultatów
Prezentacja wyników w postaci wykresów została przygotowana w notatniku Jupyter. Znajduje się on w folderze `src` pod nazwą `notebook.ipynb`. Do generacja wykresów wykorzystana została implementacja twórców SpinningUP-a, która została przerobiona pod potrzeby naszego rozwiązania. Plik z autorskimi zmianami nosi nazwę `plot.py`.