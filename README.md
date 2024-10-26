# Yandex Music FuseFS

⚠️ Это неофициальная ФС.

## Описание

Сделано только для себя, чтобы слушать музыку в MPD.

### Системные зависимости

[pyfuse3](https://pyfuse3.readthedocs.io/en/latest/install.html)

#### Debian\Ubuntu

```shell
sudo apt install pkg-config fuse3 libfuse3-dev libfuse-dev -y
```

#### Fedora

```shell
sudo dnf install pkg-config fuse3 fuse3-devel python3-devel gcc -y

```

#### user_allow_other

```shell
fusermount3: option allow_other only allowed if 'user_allow_other' is set in /etc/fuse.conf
```

### Установка

[Скачать](https://github.com/vm86/yandex-music-fusefs/releases)

``` shell
pip install yandex_fuse-*.tar.gz
```

### Начало работы

#### Запускаем

```shell
systemctl --user start yamusic-fs.service
```

Или

```shell
yamusic-fs ~/Music/Yandex/
```

#### Первый запуск

После запуска, откроется страница в браузере с QR-кодом,
который нужно отсканировать в приложении Я.Ключ.
Если авторизация прошла успешна в логах появится запись "Token saved".
И начнется синхронизация плейлиста "Мне нравится".
После завершения синхронизации в логах будет строка:
"Loaded track in like playlist .."

#### Отмонтировать

```shell
systemctl stop yamusic-fs.service --user
```

Или

```shell
fusermount -u ~/Music/Yandex
```

#### Конфигурация

~/.config/yandex-fuse.json

```json
{
  "token": "",
  "best_codec": "aac",
  "blacklist": [],
}
```

token = Токен доступа

best_codec = aac или mp3

blacklist = "Черный список" жанров для "Моя волна"

#### Логи

```shell
journalctl --user -u yamusic-fs.service --no-pager
```

Или

```shell
cat ~/.cache/yandex_fuse.log
```
