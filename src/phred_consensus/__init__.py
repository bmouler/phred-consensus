"""Phred-aware Bayesian consensus calling."""

from .core import ConsensusResult, call_consensus, majority_consensus

__all__ = ["ConsensusResult", "call_consensus", "majority_consensus"]
