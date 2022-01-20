# LL7 - PPO porównanie

Porównaj działanie różnych implementacji algorytmu PPO. Wybierz co najmniej 2 różne wersje z innych bibliotek (jedna z wersji może być implementacją własną) i przetestuj ich działanie na środowisku [HalfCheetah](https://gym.openai.com/envs/HalfCheetah-v2/).

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
Do ewaluacji modeli ze spinnup-a używamy wbudowanego modułu [`run`](https://github.com/openai/spinningup/blob/master/docs/user/running.rst#id3), który dostarcza taką funkcjonalność. Przykładowe wywołanie wygląda następująco:
```sh
python -m spinup.run [algo name] [experiment flags]
```
TODO: przykład uruchomienia

W przypadku RLlib przygotowany do tego celu został skrypt `eval_ppo_rllib_ray.py`, który to przyjmuje następujące argumenty:
* path_to_checkpoint - ścieżka do *checkpoint-u*
* path_to_config - ścieżka do pliku z konfiguracją

TODO: przykład uruchomienia
## Prezentacja rezultatów
Prezentacja wyników w postaci wykresów została przygotowana w notatniku Jupyter. Znajduje się on w folderze `src` pod nazwą `notebook.ipynb`. Do generacja wykresów wykorzystana została implementacja twórców SpinningUP-a, która została przerobiona pod potrzeby naszego rozwiązania. Plik z autorskimi zmianami nosi nazwę `plot.py`.