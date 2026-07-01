# Подключение к модели Whisper на другом ПК (с GPU)

Код на ПК без GPU отправляет семплы аудиоданных на другой ПК с GPU через WebSocket-соединение, а ПК с GPU через WebSocket-соединение возвращает распознанный текст.

---

## Использование готового Docker-контейнера hwdsl2/whisper-live-server

### Подготовка

1. Если ОС на ПК с GPU — Windows, устанавливаем WSL 2 (Windows Subsystem for Linux) на этом ПК. Для установки WSL 2 открываем командную строку от имени администратора и выполняем в ней команду ниже. После установки WSL 2 перезагрузить ПК.

   ```bash
   wsl --install
   ```

2. Если ОС на ПК с GPU — Windows, убедимся, что WSL 2 установлена, открыв командную строку от имени администратора и выполнив в ней команду ниже.

   ```bash
   wsl -l -v
   ```

3. Если ОС на ПК с GPU — Windows, нажимаем на Пуск → запускаем приложение Ubuntu.

4. Если ОС на ПК с GPU — Windows, устанавливаем Docker Desktop для Windows. Если ОС — Linux, устанавливаем Docker Desktop для Linux.

5. Если ОС на ПК с GPU — Windows, открыть Docker Desktop → Settings → General → убедиться, что на «Use the WSL 2 based engine» стоит галочка.

6. Если ОС на ПК с GPU — Windows, открыть Docker Desktop → Settings → Resources → WSL integration → поставить галочки на «Enable integration with my default WSL distro» и «Ubuntu» → нажать на Apply & restart.

7. Устанавливаем драйверы для GPU NVIDIA:

   - Если ОС на ПК с GPU — Linux, в терминале ПК выполняем команды ниже. Также есть инструкция по ссылке https://docs.nvidia.com/cuda/cuda-installation-guide-linux/, но команды в ней отличаются от команд ниже.

   ```bash
   sudo add-apt-repository ppa:graphics-drivers/ppa -y
   sudo apt update
   sudo apt install nvidia-driver-575-server
   sudo reboot
   ```

   Проверка установки:

   ```bash
   nvidia-smi
   ```

8. Устанавливаем NVIDIA Container Toolkit согласно инструкции по ссылке https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html.

   Проверка установки:

   ```bash
   nvidia-ctk --version
   ```

9. Настраиваем Docker Desktop для работы с GPU с помощью команд ниже.

   ```bash
   # Настройка рантайма Docker для использования NVIDIA Container Runtime
   sudo nvidia-ctk runtime configure --runtime=docker

   # Перезапуск Docker для применения изменений
   sudo systemctl restart docker
   ```

   Проверка конфигурации:

   ```bash
   # Проверка, что NVIDIA runtime добавлен в конфигурацию Docker
   cat /etc/docker/daemon.json
   # Должен содержать секцию "runtimes" с "nvidia"

   # Тестовый запуск GPU-контейнера
   docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
   # Должен показать информацию о GPU из контейнера
   ```

   Проверка настройки:

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

---

### Шаг 1 — Запуск Whisper в режиме сервера на ПК с GPU

Для этого:

1. Если ОС на ПК с GPU — Windows, нажимаем на Пуск → запускаем приложение Ubuntu → в нем выполняем команду ниже.

2. Если ОС на ПК с GPU — Linux, открываем командную строку и выполняем в ней команду ниже.

```bash
docker run -d \
  --name whisper-live \
  --restart=always \
  --gpus all \
  -v whisper-live-data:/var/lib/whisper-live \
  -p 9090:9090 \
  -p 8000:8000 \
  hwdsl2/whisper-live-server
```

#### Пояснение каждого параметра запуска

| Параметр | Что означает |
|----------|--------------|
| `docker run` | Запустить новый контейнер |
| `-d` | Запустить в фоновом режиме (detached mode) |
| `--name whisper-live` | Дать контейнеру имя `whisper-live` (чтобы потом обращаться к нему по имени) |
| `--restart=always` | Автоматически перезапускать контейнер, если он остановится (например, после перезагрузки ПК) |
| `--gpus all` | Разрешает контейнеру использовать все GPU, установленные на хост-системе. Если нужно только одно устройство, можно указать `--gpus device=0` |
| `-v whisper-live-data:/var/lib/whisper-live` | Создать том для хранения моделей и кэша, чтобы при обновлении контейнера не скачивать их заново |
| `-p 9090:9090` | Пробросить порт 9090 (WebSocket для потокового аудио) |
| `-p 8000:8000` | Пробросить порт 8000 (HTTP для файлов) |
| `hwdsl2/whisper-live-server` | Имя Docker-образа, из которого создаётся контейнер |

---

### Шаг 2 — Внести изменения в пользовательский код

1. В файле `config.py` атрибут `server_host` класса `Config` инициализировать IP-адресом ПК с GPU.

   **Важно:** IP-адрес должен быть статическим или вы должны знать его заранее. Узнать его на сервере можно командой ниже.

   ```bash
   hostname -I
   ```

2. В файле `context.py`, в конструкторе класса `Context` закомментировать строку

   ```python
   self.recognizer = FasterWhisperRecognizer(self.config)
   ```

   и раскомментировать строку

   ```python
   self.recognizer = RemoteWhisperRecognizer(self.config)
   ```

3. Проверить, может ли клиент достучаться до сервера, путем запуска файла `test_connection.py`.

---

**Ссылка:** https://hub.docker.com/r/hwdsl2/whisper-live-server