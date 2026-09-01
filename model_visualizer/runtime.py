"""Pomoćne funkcije za obradu i proveru example inputa."""
import torch
import torch.nn as nn

def normalizuj_ulaze(example_inputs=None, example_kwargs=None):
    """Pretvara prosleđene ulaze u args/kwargs oblik koji koriste model i torch.export."""

    if example_inputs is None:
        args = ()
    elif isinstance(example_inputs, tuple):
        args = example_inputs
    elif isinstance(example_inputs, list):
        args = tuple(example_inputs)
    else:
        args = (example_inputs,)

    if example_kwargs is None:
        kwargs = {}
    elif isinstance(example_kwargs, dict):
        kwargs = dict(example_kwargs)
    else:
        raise TypeError('example_kwargs mora biti dict ili None.')

    return (args, kwargs)

def opisi_vrednost(value):
    """Pravi kratak tekstualni opis ulaza, prvenstveno za poruke o grešci."""

    if isinstance(value, torch.Tensor):
        return f'Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})'

    if isinstance(value, tuple):
        return '(' + ', '.join((opisi_vrednost(item) for item in value)) + ')'

    if isinstance(value, list):
        return '[' + ', '.join((opisi_vrednost(item) for item in value)) + ']'

    if isinstance(value, dict):
        content = ', '.join((f'{key}: {opisi_vrednost(item)}' for key, item in value.items()))
        return '{' + content + '}'

    return repr(value)

def opisi_ulaze(args, kwargs):

    lines = []

    if args:
        lines.append('Pozicioni argumenti:')
        for index, value in enumerate(args):
            lines.append(f'  [{index}] {opisi_vrednost(value)}')

    if kwargs:
        lines.append('Keyword argumenti:')
        for key, value in kwargs.items():
            lines.append(f'  {key}={opisi_vrednost(value)}')

    if not lines:
        return 'Nisu prosleđeni example input-i.'

    return '\n'.join(lines)

def proveri_ulaze_modela(model, example_inputs=None, example_kwargs=None):
    """Jednim probnim pozivom proverava da li prosleđeni ulazi odgovaraju modelu."""

    if not isinstance(model, nn.Module):
        raise TypeError('Očekivan je PyTorch model tipa torch.nn.Module.')

    args, kwargs = normalizuj_ulaze(example_inputs, example_kwargs)

    if not args and (not kwargs):
        return

    was_training = model.training

    try:
        model.eval()
        with torch.no_grad():
            model(*args, **kwargs)
    except Exception as error:
        description = opisi_ulaze(args, kwargs)
        raise ValueError(f'Prosleđeni example input-i nisu kompatibilni sa modelom.\n\n{description}\n\nOriginalna greška: {type(error).__name__}: {error}') from error
    finally:
        model.train(was_training)
