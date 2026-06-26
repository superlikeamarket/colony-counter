from src.corrections.correction_manager import CorrectionManager

manager = CorrectionManager("local/corrected/corrections")

state = manager.create_state_from_predictions(
    image_id="test_plate",
    centers=[(100, 100), (200, 200), (300, 300)],
    sample_type="petri_dish",
)

state = manager.add_point(state, 400, 400)
state = manager.remove_nearest_detection(state, 100, 100)

print("Model count:", state.model_count())
print("Final count:", state.final_count())

manager.save_state(state)

loaded_state = manager.load_state("test_plate")
print("Loaded final count:", loaded_state.final_count())