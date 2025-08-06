import traceback

print("--- Starting Hailo Library Inspection ---")

try:
    # Use the import that we know works from previous diagnostics
    from hailo_platform import VDevice, HEF
    print("[INFO] Import of VDevice and HEF successful.")

    # Assume the .hef file is in the current directory
    model_path = "yolov8m.hef" 
    print(f"[INFO] Attempting to load model: {model_path}")

    with VDevice() as vdevice:
        print("[INFO] VDevice initialized successfully.")
        
        hef = HEF(model_path)
        print("[INFO] HEF object created successfully.")
        
        network_groups = vdevice.configure(hef)
        network_group = network_groups[0]
        print("[INFO] VDevice configured successfully.")

        with network_group.activate() as activated_group:
            print("[SUCCESS] Network Group activated. Now inspecting the object...")
            print("\n" + "="*60)
            print("--- Attributes and Methods available on 'activated_group' ---")
            
            # This is the most important line. It lists everything we can do.
            print(dir(activated_group))
            
            print("="*60 + "\n")
            print("--- Inspection Complete ---")
            print("Please copy the list of names above (inside the ===== lines) and provide it.")

except FileNotFoundError:
    print(f"\n[FATAL] The model file '{model_path}' was not found in this directory.")
    print("       Please ensure your yolov8m.hef file is present.")
except Exception as e:
    print(f"\n[FATAL] An error occurred during inspection: {e}")
    traceback.print_exc()