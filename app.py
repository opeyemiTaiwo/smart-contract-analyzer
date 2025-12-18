"""
AI-Driven Smart Contract Reentrancy Detection
Production Application - Trained Model Only
Author: Opeyemi
Institution: Morgan State University
"""

import gradio as gr
import pandas as pd
import numpy as np
import re
import joblib
import os
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_reentrancy_features(source_code):
    """
    Extract 38 reentrancy-specific features from smart contract source code
    These features match exactly what the model was trained on
    """
    features = {}
    code = str(source_code).lower()

    # === EXTERNAL CALL PATTERNS ===
    features['call_value_count'] = len(re.findall(r'\.call\s*\{\s*value\s*:', code))
    features['call_count'] = len(re.findall(r'\.call\(', code))
    features['transfer_count'] = len(re.findall(r'\.transfer\(', code))
    features['send_count'] = len(re.findall(r'\.send\(', code))
    features['delegatecall_count'] = len(re.findall(r'\.delegatecall\(', code))

    # === STATE CHANGE PATTERNS ===
    features['state_after_call'] = int(bool(re.search(
        r'(\.call|\.transfer|\.send).*?[\n\r].*?(\w+\s*=|\w+\s*\+=|\w+\s*-=)', code, re.DOTALL
    )))

    # === REENTRANCY PROTECTION ===
    features['has_nonreentrant'] = int('nonreentrant' in code)
    features['has_mutex'] = int('mutex' in code or 'locked' in code)
    features['has_reentrancy_guard'] = int(features['has_nonreentrant'] or features['has_mutex'])

    # === BALANCE PATTERNS ===
    features['balance_keyword'] = len(re.findall(r'\bbalance\b', code))
    features['balance_mapping'] = len(re.findall(r'balance\s*\[', code))
    features['withdraw_function'] = int('function withdraw' in code or 'function _withdraw' in code)
    features['withdrawal_pattern'] = len(re.findall(r'withdraw', code))

    # === FUNCTION TYPES ===
    features['fallback_function'] = int('fallback' in code)
    features['receive_function'] = int('receive' in code)
    features['payable_functions'] = len(re.findall(r'function\s+\w+.*?payable', code))

    # === ACCESS CONTROL ===
    features['require_count'] = len(re.findall(r'\brequire\s*\(', code))
    features['assert_count'] = len(re.findall(r'\bassert\s*\(', code))
    features['modifier_count'] = len(re.findall(r'\bmodifier\s+\w+', code))
    features['onlyowner'] = int('onlyowner' in code or 'only_owner' in code)

    # === CHECKS-EFFECTS-INTERACTIONS ===
    features['checks_before_effects'] = len(re.findall(r'require.*?=.*?call', code, re.DOTALL))

    # === LOOPS ===
    features['for_loop_count'] = len(re.findall(r'\bfor\s*\(', code))
    features['while_loop_count'] = len(re.findall(r'\bwhile\s*\(', code))

    # === EXTERNAL INTERACTIONS ===
    features['interface_count'] = len(re.findall(r'\binterface\s+\w+', code))
    features['import_count'] = len(re.findall(r'\bimport\s+', code))
    features['external_keyword'] = len(re.findall(r'\bexternal\b', code))

    # === DANGEROUS PATTERNS ===
    features['selfdestruct'] = int('selfdestruct' in code)
    features['suicide'] = int('suicide' in code)

    # === CODE STRUCTURE ===
    features['function_count'] = len(re.findall(r'\bfunction\s+\w+', code))
    features['event_count'] = len(re.findall(r'\bevent\s+\w+', code))
    features['mapping_count'] = len(re.findall(r'\bmapping\s*\(', code))
    features['struct_count'] = len(re.findall(r'\bstruct\s+\w+', code))

    # === GAS PATTERNS ===
    features['gas_limit'] = int('gas:' in code or '.gas(' in code)
    features['gas_left'] = int('gasleft()' in code)

    # === VULNERABLE COMBINATIONS ===
    features['vulnerable_combo_1'] = int(
        (features['call_value_count'] > 0 or features['call_count'] > 0) and
        features['has_reentrancy_guard'] == 0 and
        features['state_after_call'] > 0
    )
    
    features['vulnerable_combo_2'] = int(
        features['withdraw_function'] > 0 and
        features['has_reentrancy_guard'] == 0
    )

    # === SECURITY CHECK ===
    features['security_check'] = int('require' in code or 'assert' in code or 'revert' in code)

    # === CODE LENGTH ===
    features['sourcecode_len'] = len(source_code)
    features['bytecode_len'] = len(source_code) // 2

    return features


def preprocess_code(source_code):
    """
    Clean and preprocess Solidity code for TF-IDF vectorization
    Must match exactly what was done during training
    """
    code = str(source_code).lower()
    
    # Remove single-line comments
    code = re.sub(r'//.*?\n', ' ', code)
    
    # Remove multi-line comments
    code = re.sub(r'/\*.*?\*/', ' ', code, flags=re.DOTALL)
    
    # Remove string literals
    code = re.sub(r'"[^"]*"', ' ', code)
    code = re.sub(r"'[^']*'", ' ', code)
    
    # Remove numbers
    code = re.sub(r'\b\d+\b', ' ', code)
    
    # Keep only alphanumeric characters and underscores
    code = re.sub(r'[^a-z0-9_\s]', ' ', code)
    
    # Remove extra whitespace
    code = ' '.join(code.split())
    
    return code


# ============================================================================
# LOAD TRAINED MODELS (REQUIRED)
# ============================================================================

print("Loading trained models...")

try:
    # Load all model components
    model = joblib.load('model/reentrancy_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    tfidf = joblib.load('model/tfidf_vectorizer.pkl')
    feature_info = joblib.load('model/feature_info.pkl')
    
    # Get feature names
    structural_features = feature_info['structural_features']
    
    print("✅ All models loaded successfully!")
    print(f"   - Model: {type(model).__name__}")
    print(f"   - Structural features: {len(structural_features)}")
    print(f"   - TF-IDF features: {feature_info['n_tfidf']}")
    print(f"   - Total features: {feature_info['n_structural'] + feature_info['n_tfidf']}")
    
    MODELS_LOADED = True

except Exception as e:
    print(f"❌ ERROR: Could not load models!")
    print(f"   Error: {str(e)}")
    print(f"\n   Make sure you have:")
    print(f"   1. Created model/ folder")
    print(f"   2. Uploaded these files to model/:")
    print(f"      - reentrancy_model.pkl")
    print(f"      - scaler.pkl")
    print(f"      - tfidf_vectorizer.pkl")
    print(f"      - feature_info.pkl")
    
    MODELS_LOADED = False
    model = None
    scaler = None
    tfidf = None
    structural_features = None


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_reentrancy(source_code):
    """
    Predict if a smart contract has reentrancy vulnerability
    Uses trained Random Forest model with 95%+ accuracy
    """
    
    # Validate model is loaded
    if not MODELS_LOADED:
        return {
            "❌ Error": "Models not loaded",
            "Help": "Please check server logs. Model files may be missing.",
            "Required Files": "model/reentrancy_model.pkl, model/scaler.pkl, model/tfidf_vectorizer.pkl, model/feature_info.pkl"
        }
    
    # Validate input
    if not source_code or len(source_code.strip()) < 10:
        return {
            "⚠️ Error": "Please provide valid Solidity source code",
            "Minimum Length": "At least 10 characters required"
        }
    
    try:
        # ========================================================================
        # STEP 1: Extract structural features
        # ========================================================================
        features_dict = extract_reentrancy_features(source_code)
        feature_df = pd.DataFrame([features_dict])
        
        # Ensure all expected columns exist and are in correct order
        for col in structural_features:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        # Reorder to match training exactly
        feature_df = feature_df[structural_features]
        X_structural = feature_df.values
        
        # ========================================================================
        # STEP 2: Scale structural features
        # ========================================================================
        X_structural_scaled = scaler.transform(X_structural)
        
        # ========================================================================
        # STEP 3: Extract text features
        # ========================================================================
        processed_code = preprocess_code(source_code)
        X_tfidf = tfidf.transform([processed_code])
        
        # ========================================================================
        # STEP 4: Combine features
        # ========================================================================
        X_combined = hstack([X_structural_scaled, X_tfidf])
        
        # ========================================================================
        # STEP 5: Make prediction
        # ========================================================================
        prediction = model.predict(X_combined)[0]
        probabilities = model.predict_proba(X_combined)[0]
        
        # Get confidence score (probability of predicted class)
        confidence = probabilities[prediction]
        
        # Get reentrancy probability
        reentrancy_prob = probabilities[1]
        
        # ========================================================================
        # STEP 6: Analyze and create detailed report
        # ========================================================================
        
        # Determine risk level
        if reentrancy_prob >= 0.7:
            risk_level = "🔴 HIGH RISK"
            risk_description = "Reentrancy vulnerability highly likely"
        elif reentrancy_prob >= 0.4:
            risk_level = "🟡 MEDIUM RISK"
            risk_description = "Potential vulnerability detected"
        else:
            risk_level = "🟢 LOW RISK"
            risk_description = "Contract appears relatively safe"
        
        # Count critical issues
        critical_issues = []
        if features_dict['call_count'] > 0 or features_dict['call_value_count'] > 0:
            if features_dict['has_reentrancy_guard'] == 0:
                critical_issues.append("External calls without reentrancy protection")
        
        if features_dict['state_after_call'] > 0:
            critical_issues.append("State changes detected after external calls")
        
        if features_dict['vulnerable_combo_1']:
            critical_issues.append("Dangerous pattern: External call + No guard + State change")
        
        if features_dict['withdraw_function'] and features_dict['has_reentrancy_guard'] == 0:
            critical_issues.append("Withdrawal function without protection")
        
        # Count security features
        security_features = []
        if features_dict['has_reentrancy_guard']:
            security_features.append("Reentrancy guard present")
        
        if features_dict['require_count'] > 0:
            security_features.append(f"{features_dict['require_count']} require statements")
        
        if features_dict['modifier_count'] > 0:
            security_features.append(f"{features_dict['modifier_count']} custom modifiers")
        
        # Generate recommendations
        recommendations = []
        if not features_dict['has_reentrancy_guard'] and (features_dict['call_count'] > 0 or features_dict['call_value_count'] > 0):
            recommendations.append("⚠️ Add nonReentrant modifier to functions with external calls")
        
        if features_dict['state_after_call']:
            recommendations.append("⚠️ Follow Checks-Effects-Interactions pattern: update state before external calls")
        
        if features_dict['withdraw_function'] and not features_dict['has_reentrancy_guard']:
            recommendations.append("⚠️ Protect withdrawal functions with mutex or nonReentrant modifier")
        
        if features_dict['require_count'] == 0 and features_dict['assert_count'] == 0:
            recommendations.append("⚠️ Add input validation with require() statements")
        
        if not recommendations:
            recommendations.append("✓ Contract follows good security practices")
            recommendations.append("✓ Continue following secure coding patterns")
        
        # ========================================================================
        # STEP 7: Format result
        # ========================================================================
        
        result = {
            "🎯 Risk Assessment": f"{risk_level} - {risk_description}",
            "📊 Reentrancy Probability": f"{reentrancy_prob:.1%}",
            "🔒 Model Confidence": f"{confidence:.1%}",
            "": "",
            "📋 Analysis Summary": "",
            "Critical Issues": f"{len(critical_issues)} found" if critical_issues else "None detected ✓",
            "Security Features": f"{len(security_features)} present" if security_features else "None detected",
            " ": "",
            "🔍 Detailed Findings": "",
            "External Calls (.call/.transfer/.send)": f"{'⚠️' if features_dict['call_count'] > 0 or features_dict['call_value_count'] > 0 else '✓'} {features_dict['call_count'] + features_dict['call_value_count']} detected",
            "Reentrancy Guards": f"{'✓ Protected' if features_dict['has_reentrancy_guard'] else '⚠️ Not Protected'}",
            "State Changes After External Calls": f"{'⚠️ Detected' if features_dict['state_after_call'] else '✓ Safe'}",
            "Withdrawal Functions": f"{features_dict['withdraw_function']} found",
            "Access Controls (require/assert)": f"{features_dict['require_count'] + features_dict['assert_count']} checks",
            "Function Modifiers": f"{features_dict['modifier_count']} custom modifiers",
            "  ": "",
        }
        
        # Add critical issues if any
        if critical_issues:
            result["⚠️ Critical Issues Detected"] = ""
            for i, issue in enumerate(critical_issues, 1):
                result[f"   {i}"] = issue
            result["   "] = ""
        
        # Add security features if any
        if security_features:
            result["✅ Security Features Found"] = ""
            for i, feature in enumerate(security_features, 1):
                result[f"    {i}"] = feature
            result["    "] = ""
        
        # Add recommendations
        result["💡 Recommendations"] = ""
        for i, rec in enumerate(recommendations, 1):
            result[f"     {i}"] = rec
        
        return result
        
    except Exception as e:
        return {
            "❌ Analysis Failed": str(e),
            "Error Type": type(e).__name__,
            "Help": "Please ensure you provided valid Solidity source code",
            "Debug Info": "Check that all model files are loaded correctly"
        }


# ============================================================================
# SAMPLE CONTRACTS FOR TESTING
# ============================================================================

SAMPLE_VULNERABLE = """pragma solidity ^0.8.0;

contract VulnerableBank {
    mapping(address => uint) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint _amount) public {
        require(balances[msg.sender] >= _amount, "Insufficient balance");
        
        // VULNERABILITY: External call before state update
        (bool sent, ) = msg.sender.call{value: _amount}("");
        require(sent, "Failed to send Ether");
        
        // State change AFTER external call - allows reentrancy!
        balances[msg.sender] -= _amount;
    }
    
    function getBalance() public view returns (uint) {
        return balances[msg.sender];
    }
}
"""

SAMPLE_SAFE = """pragma solidity ^0.8.0;

contract SafeBank {
    mapping(address => uint) public balances;
    bool private locked;
    
    modifier nonReentrant() {
        require(!locked, "No reentrancy allowed");
        locked = true;
        _;
        locked = false;
    }

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint _amount) public nonReentrant {
        require(balances[msg.sender] >= _amount, "Insufficient balance");
        
        // SAFE: State change BEFORE external call
        balances[msg.sender] -= _amount;
        
        // External call after state update
        (bool sent, ) = msg.sender.call{value: _amount}("");
        require(sent, "Failed to send Ether");
    }
    
    function getBalance() public view returns (uint) {
        return balances[msg.sender];
    }
}
"""


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create production Gradio interface"""
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Smart Contract Security Analyzer",
        css="""
        .gradio-container {max-width: 1200px !important}
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 🔐 Smart Contract Security Analyzer
            ## AI-Driven Reentrancy Vulnerability Detection
            
            Analyze Solidity smart contracts for reentrancy vulnerabilities using our trained Random Forest model with **95%+ accuracy**.
            
            **Model:** Random Forest | **Features:** 338 (38 structural + 300 TF-IDF) | **Trained on:** 40,000+ contracts
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                code_input = gr.Code(
                    label="📝 Solidity Smart Contract Code",
                    language="solidity",
                    lines=25,
                    value=SAMPLE_VULNERABLE
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Contract", variant="primary", size="lg", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", size="lg", scale=1)
                
                with gr.Row():
                    sample_vuln_btn = gr.Button("⚠️ Vulnerable Example", size="sm")
                    sample_safe_btn = gr.Button("✅ Safe Example", size="sm")
            
            with gr.Column(scale=2):
                output = gr.JSON(label="🎯 Security Analysis Results")
        
        gr.Markdown(
            """
            ---
            ### 📚 About Reentrancy Vulnerabilities
            
            Reentrancy occurs when a contract makes an external call before updating its state, allowing attackers to recursively call back and drain funds.
            
            **Famous Example:** The DAO Hack (2016) - $60M stolen
            
            ### 🛡️ Best Practices
            ✅ Use reentrancy guards (`nonReentrant` modifier)  
            ✅ Follow CEI pattern (Checks-Effects-Interactions)  
            ✅ Update state before external calls
            
            ---
            **Developed by:** Opeyemi | **Institution:** Morgan State University | **Research:** AI-Based Smart Contract Security
            """
        )
        
        # Button actions
        analyze_btn.click(fn=predict_reentrancy, inputs=code_input, outputs=output)
        clear_btn.click(fn=lambda: "", outputs=code_input)
        sample_vuln_btn.click(fn=lambda: SAMPLE_VULNERABLE, outputs=code_input)
        sample_safe_btn.click(fn=lambda: SAMPLE_SAFE, outputs=code_input)
    
    return demo


# ============================================================================
# LAUNCH APPLICATION
# ============================================================================

if __name__ == "__main__":
    if not MODELS_LOADED:
        print("\n" + "="*70)
        print("⚠️  WARNING: Models not loaded!")
        print("="*70)
        print("Application will start but predictions will fail.")
        print("Upload model files to model/ folder.\n")
    
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)