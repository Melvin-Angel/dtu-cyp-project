"""Pretrained Atom-JEPA encoder with a readout head from models/readouts.py."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from atom_jepa import AtomJEPA
from models.readouts import build_readout


class Conformers(Dataset):
    """One item per molecule, or one per (molecule, conformer) pair when repeats > 1.

    Conformer files store positions as [conformers, atoms, 3]; files written before
    multi-conformer preparation store [atoms, 3] and are read as a single conformer.
    With sample=True a random conformer is drawn each time a molecule is accessed."""

    def __init__(self, frame, folder, targets, sample=False, repeats=1):
        self.frame, self.folder = frame.reset_index(drop=True), folder
        self.y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)
        self.sample, self.repeats = sample, repeats

    def __len__(self):
        return len(self.frame) * self.repeats

    def __getitem__(self, i):
        # Molecule-major order, so fold_predictions can reshape eval output directly.
        row, slot = divmod(i, self.repeats)
        with np.load(self.folder / f'{self.frame.iloc[row].key}.npz') as d:
            z, pos = d['z'], d['pos']
            if pos.ndim == 2:
                pos = pos[None]
            k = int(np.random.randint(len(pos))) if self.sample else slot % len(pos)
            return {'atomic_numbers': torch.tensor(z, dtype=torch.long),
                    'node_coordinates': torch.tensor(pos[k], dtype=torch.float32),
                    'y': torch.tensor(self.y[row])}


class AtomJEPAModel(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()
        self.cfg, self.folder = cfg, data_dir / 'conformers'
        self.encoder = AtomJEPA.from_pretrained(cfg.checkpoint, device=device,
                                                precision=cfg.precision, cuequivariance=False)
        self.encoder.set_grad_checkpointing(not cfg.freeze_encoder)
        self.encoder.requires_grad_(not cfg.freeze_encoder)
        self.readout = build_readout(readout, self.encoder.embedding_dim, n_targets).to(self.encoder.device)
        # Conformer handling; the defaults reproduce single-conformer behaviour.
        self.sample_conformers = bool(cfg.get('sample_conformers', False))
        self.eval_conformers = int(cfg.get('eval_conformers', 1))
        if self.eval_conformers < 1:
            raise ValueError('model.eval_conformers must be at least 1')

    @property
    def device(self):
        return self.encoder.device

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.freeze_encoder:
            self.encoder.eval()
        return self

    def fit_preprocessing(self, train):
        pass

    def loader(self, frame, targets, batch_size, shuffle=False):
        # shuffle=True is only ever used for the training loader, so it marks train vs eval.
        repeats = 1 if shuffle else self.eval_conformers
        dataset = Conformers(frame, self.folder, targets,
                             sample=shuffle and self.sample_conformers, repeats=repeats)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=self.encoder.collate, num_workers=0)

    def fold_predictions(self, values):
        """Average the eval_conformers predictions made for each molecule."""
        if self.eval_conformers == 1:
            return values
        return values.reshape(-1, self.eval_conformers, values.shape[-1]).mean(axis=1)

    def param_groups(self):
        groups = [{'params': list(self.readout.parameters()), 'lr': self.cfg.readout_lr}]
        if not self.cfg.freeze_encoder:
            groups.append({'params': list(self.encoder.parameters()), 'lr': self.cfg.encoder_lr})
        return groups

    def forward(self, batch):
        return self.readout(self.encoder(batch))  # Differentiable forward; do NOT use encoder.embed().
