"""
Test script to verify the implementation is working correctly
"""

import torch
import sys

def test_imports():
    """Test if all imports work"""
    print("Testing imports...")

    try:
        from probabilistic_models import ProbabilisticFIN, DeterministicFIN, BetaNLLLoss
        print("✓ probabilistic_models imported successfully")
    except Exception as e:
        print(f"✗ Error importing probabilistic_models: {e}")
        return False

    try:
        from multimodal_model import get_model, ImageEncoder, TextEncoder
        print("✓ multimodal_model imported successfully")
    except Exception as e:
        print(f"✗ Error importing multimodal_model: {e}")
        return False

    try:
        from federated_learning import FederatedTrainer, fedavg_aggregate
        print("✓ federated_learning imported successfully")
    except Exception as e:
        print(f"✗ Error importing federated_learning: {e}")
        return False

    try:
        from data_loader_federated import create_federated_dataloaders, create_test_dataloader
        print("✓ data_loader_federated imported successfully")
    except Exception as e:
        print(f"✗ Error importing data_loader_federated: {e}")
        return False

    return True


def test_probabilistic_fin():
    """Test Probabilistic FIN"""
    print("\nTesting Probabilistic FIN...")

    from probabilistic_models import ProbabilisticFIN, BetaNLLLoss

    try:
        # Create model
        model = ProbabilisticFIN(input_dim=256, output_dim=256)

        # Test forward pass
        batch_size = 4
        source_features = torch.randn(batch_size, 256)

        alpha, beta, mean = model(source_features)

        assert alpha.shape == (batch_size, 256), f"Alpha shape mismatch: {alpha.shape}"
        assert beta.shape == (batch_size, 256), f"Beta shape mismatch: {beta.shape}"
        assert mean.shape == (batch_size, 256), f"Mean shape mismatch: {mean.shape}"

        # Test uncertainty
        uncertainty = model.get_uncertainty(alpha, beta)
        assert uncertainty.shape == (batch_size, 256), f"Uncertainty shape mismatch"

        # Test loss
        loss_fn = BetaNLLLoss()
        target = torch.rand(batch_size, 256)
        loss = loss_fn(alpha, beta, target)

        assert not torch.isnan(loss), "Loss is NaN!"
        assert not torch.isinf(loss), "Loss is inf!"

        print(f"✓ Probabilistic FIN working correctly")
        print(f"  Alpha range: [{alpha.min():.3f}, {alpha.max():.3f}]")
        print(f"  Beta range: [{beta.min():.3f}, {beta.max():.3f}]")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Mean uncertainty: {uncertainty.mean().item():.6f}")

        return True

    except Exception as e:
        print(f"✗ Error in Probabilistic FIN: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multimodal_model():
    """Test Multimodal Model"""
    print("\nTesting Multimodal Model...")

    try:
        from multimodal_model import get_model

        # Create model (without pretrained weights for speed)
        model = get_model(use_uncertainty=True, pretrained=False)

        # Test forward pass
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        texts = ["This is a test report.", "Another medical report."]

        img_feat, txt_feat, outputs = model(images, texts)

        assert img_feat.shape == (batch_size, 256), f"Image features shape mismatch"
        assert txt_feat.shape == (batch_size, 256), f"Text features shape mismatch"
        assert outputs['logits'].shape == (batch_size, 14), f"Logits shape mismatch"

        if 'alpha' in outputs:
            assert outputs['alpha'].shape == (batch_size, 14), f"Alpha shape mismatch"
            assert outputs['beta'].shape == (batch_size, 14), f"Beta shape mismatch"

        print(f"✓ Multimodal Model working correctly")
        print(f"  Image features: {img_feat.shape}")
        print(f"  Text features: {txt_feat.shape}")
        print(f"  Logits: {outputs['logits'].shape}")

        return True

    except Exception as e:
        print(f"✗ Error in Multimodal Model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_federated_aggregation():
    """Test Federated Aggregation"""
    print("\nTesting Federated Aggregation...")

    try:
        from multimodal_model import get_model
        from probabilistic_models import ProbabilisticFIN
        from federated_learning import FederatedTrainer
        import copy

        # Create models
        global_model = get_model(pretrained=False)
        global_fin_i2t = ProbabilisticFIN()
        global_fin_t2i = ProbabilisticFIN()

        # Create trainer
        trainer = FederatedTrainer(
            global_model=global_model,
            global_fin_i2t=global_fin_i2t,
            global_fin_t2i=global_fin_t2i,
            num_clients=10,
            use_uq_weighting=True
        )

        # Distribute models
        client_models, client_fins_i2t, client_fins_t2i = trainer.distribute_models()

        assert len(client_models) == 10, "Wrong number of client models"
        assert len(client_fins_i2t) == 10, "Wrong number of FINs"

        # Simulate training with different uncertainties
        for i in range(10):
            trainer.update_client_stats(
                client_id=i,
                uncertainty=0.01 + i * 0.01,
                data_size=1000
            )

        # Aggregate
        multimodal_indices = [0, 1, 2, 3, 4]
        trainer.aggregate_round(
            client_models,
            client_fins_i2t,
            client_fins_t2i,
            multimodal_indices
        )

        print(f"✓ Federated Aggregation working correctly")

        return True

    except Exception as e:
        print(f"✗ Error in Federated Aggregation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("TESTING PROBABILISTIC FEDFEATGEN IMPLEMENTATION")
    print("="*60)

    all_passed = True

    # Test imports
    if not test_imports():
        print("\n✗ Import test failed!")
        all_passed = False
        return

    # Test Probabilistic FIN
    if not test_probabilistic_fin():
        print("\n✗ Probabilistic FIN test failed!")
        all_passed = False

    # Test Multimodal Model
    if not test_multimodal_model():
        print("\n✗ Multimodal Model test failed!")
        all_passed = False

    # Test Federated Aggregation
    if not test_federated_aggregation():
        print("\n✗ Federated Aggregation test failed!")
        all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nYou can now run training with:")
        print("  python train_federated.py --use_probabilistic --use_uq_weighting")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the errors above before running training.")
    print("="*60)


if __name__ == "__main__":
    main()
