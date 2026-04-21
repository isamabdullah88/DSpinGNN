# ==========================================
# 1. Directory Scanning Module
# ==========================================
class DirectoryExplorer:
    """Scans the dataset directory and yields clean, valid sample paths."""
    @staticmethod
    def find_samples(dataset_dir, phase='FM'):
        base_path = Path(dataset_dir)
        samples = []
        
        if not base_path.exists():
            return samples

        # Assumed structure: base_dir / StrainType / FM / Strain_Type_Value_RattleIdx
        for stn_dir in base_path.iterdir():
            if stn_dir.is_dir() and not stn_dir.name.startswith("."):
                phase_dir = stn_dir / phase
                if phase_dir.exists():
                    for sample_dir in phase_dir.iterdir():
                        if sample_dir.is_dir() and sample_dir.name.startswith("Strain_"):
                            samples.append(sample_dir)
                            
        return sorted(samples)