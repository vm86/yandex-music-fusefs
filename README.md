## Yandex Music FuseFS

⚠️ Это неофициальная ФС.

## Описание

Cделанно только для себя, чтобы слушать музыку в MPD.

### Установка

``` shell
git clone https://github.com/vm86/yandex-music-fusefs
cd yandex-music-fusefs
pip install .
```

### Начало работы

Создать файл ~/.config/yandex-fuse.json

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

#### Токен

Задача по получению токена для доступа к данным лежит на плечах пользователя, использующих данную библиотеку. О том как получить токен читайте в [документации](https://github.com/MarshalX/yandex-music-api/blob/main/README.md#%D0%B4%D0%BE%D1%81%D1%82%D1%83%D0%BF-%D0%BA-%D0%B2%D0%B0%D1%88%D0%B8%D0%BC-%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%BC-%D1%8F%D0%BD%D0%B4%D0%B5%D0%BA%D1%81%D0%BC%D1%83%D0%B7%D1%8B%D0%BA%D0%B0).

#### Запускаем

```shell
yamusic-fs ~/Music/Yandex/
```

#### Отмонтировать

```shell
fusermount -u ~/Music/Yandex

```

Логи можно посмотреть в ~/.cache/yandex_fuse.log
