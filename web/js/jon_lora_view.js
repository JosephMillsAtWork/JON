import { app } from "/scripts/app.js";

let CACHED_LORA_NAMES = [];

app.registerExtension({
    name: "JosephOddNodes.LoRAChain",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "JonLoRAChain") {

            const resp = await fetch("/object_info/LoraLoader");
            const data = await resp.json();
            if (data && data.LoraLoader && data.LoraLoader.input.required.lora_name) {
                CACHED_LORA_NAMES = data.LoraLoader.input.required.lora_name[0];
            }

            const getCleanLabel = (path) => {
                if (!path || path === "Select LoRA...") return "Strength";
                let parts = path.split("/");
                let filename = parts[parts.length - 1];
                let clean = filename.replace(/\.[^/.]+$/, "");
                return clean + "_strength";
            };

            const updateHiddenConfig = (node) => {
                const state = {};
                if (node.widgets) {
                    node.widgets.forEach(w => {
                        if (w.name) {
                            if (w.name.startsWith("lora_name_")) {
                                const id = w.name.split("lora_name_")[1];
                                if (!state[id]) state[id] = {};
                                state[id].name = w.value;
                                if (node.inputs) {
                                    const input = node.inputs.find(i => i.loraId === id);
                                    if (input) state[id].input_name = input.name;
                                }
                            }
                            if (w.name.startsWith("lora_enabled_")) {
                                const id = w.name.split("lora_enabled_")[1];
                                if (!state[id]) state[id] = {};
                                state[id].enabled = w.value;
                            }
                        }
                    });
                }
                const configW = node.widgets.find(w => w.name === "lora_stack_json");
                if (configW) configW.value = JSON.stringify(state);
            };

                const resizeNodeKeepingWidth = (node) => {
                    const currentWidth = node.size[0];
                    const minSize = node.computeSize();
                    node.setSize([Math.max(currentWidth, minSize[0]), minSize[1]]);
                };

                const createSeparator = () => {
                    return {
                        type: "separator", name: "separator",
                        draw: function(ctx, node, w, y) {
                            ctx.strokeStyle = "#444"; ctx.beginPath(); ctx.moveTo(0, y+5); ctx.lineTo(w, y+5); ctx.stroke();
                        },
                        computeSize: (w) => [w, 10]
                    };
                };

                const addLoRASlot = function(node, index, savedName, savedEnabled) {
                    const id = index;
                    const widgetName = `lora_name_${id}`;
                    const toggleName = `lora_enabled_${id}`;
                    const initialLabel = getCleanLabel(savedName || CACHED_LORA_NAMES[0]);

                    let existingInput = null;
                    if (node.inputs) {
                        existingInput = node.inputs.find(i => i.name === initialLabel && !i.loraId);
                    }

                    if (existingInput) {
                        existingInput.loraId = id;
                    } else {
                        node.addInput(initialLabel, "FLOAT");
                        const newVal = node.inputs[node.inputs.length - 1];
                        newVal.loraId = id;
                    }

                    const sepTop = createSeparator();
                    node.widgets.push(sepTop);

                    // --- TOGGLE ---
                    const isEnabled = (savedEnabled !== undefined) ? savedEnabled : true;
                    const toggle = node.addWidget("toggle", "Enable", isEnabled, (value) => {
                        setSlotState(value);
                        updateHiddenConfig(node);
                    });
                    toggle.name = "enabled";

                    // --- COMBO ---
                    const combo = node.addWidget("combo", widgetName, savedName || CACHED_LORA_NAMES[0], (value) => {
                        const newLabel = getCleanLabel(value);
                        const targetInput = node.inputs.find(i => i.loraId === id);
                        if (targetInput) targetInput.name = newLabel;
                        updateHiddenConfig(node);
                    }, { values: CACHED_LORA_NAMES });
                    combo.label = "lora_name";

                    // --- REMOVE BUTTON ---
                    const btn = node.addWidget("button", `❌ Remove`, null, () => {
                        const inputIdx = node.inputs.findIndex(i => i.loraId === id);
                        if (inputIdx > -1) node.removeInput(inputIdx);

                        [sepTop, toggle, combo, btn, sepBottom].forEach(w => {
                            const idx = node.widgets.indexOf(w);
                            if (idx > -1) node.widgets.splice(idx, 1);
                        });

                            updateHiddenConfig(node);
                            resizeNodeKeepingWidth(node);
                            node.setDirtyCanvas(true, true);
                    });

                    const sepBottom = createSeparator();
                    node.widgets.push(sepBottom);

                    // --- VISUAL STATE HELPER ---
                    const setSlotState = (enabled) => {
                        // 1. Disable the Dropdown (So you can't change it while off)
                        combo.disabled = !enabled;

                        // 2. Keep Remove Button ACTIVE (So you can delete it even if disabled)
                        btn.disabled = false;

                        // 3. Keep Toggle ACTIVE (So you can turn it back on!)
                        toggle.disabled = false;
                    };

                    // Apply initial state
                    setSlotState(isEnabled);

                    return { sepTop, toggle, combo, btn, sepBottom };
                };

                const onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function () {
                    const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                    const configWidget = this.widgets ? this.widgets.find(w => w.name === "lora_stack_json") : null;
                    if (configWidget) {
                        configWidget.type = "hidden";
                        configWidget.computeSize = () => [0,0];
                    }
                    const jsonInputIdx = this.findInputSlot("lora_stack_json");
                    if (jsonInputIdx > -1) this.removeInput(jsonInputIdx);

                    this.addWidget("button", "Add LoRA", null, () => {
                        const uniqueID = Date.now().toString().slice(-6);
                        addLoRASlot(this, uniqueID, null, true);
                        resizeNodeKeepingWidth(this);
                        updateHiddenConfig(this);
                    });

                    return r;
                };

                const onConfigure = nodeType.prototype.onConfigure;
                nodeType.prototype.onConfigure = function(w) {
                    if (onConfigure) onConfigure.apply(this, arguments);

                    let state = {};
                    if (w.widgets_values && w.widgets_values.length > 0) {
                        try {
                            state = JSON.parse(w.widgets_values[0]);
                        } catch (e) { state = {}; }
                    }

                    for (const [id, data] of Object.entries(state)) {
                        addLoRASlot(this, id, data.name, data.enabled);
                    }

                    if (this.inputs) {
                        for (const [id, data] of Object.entries(state)) {
                            const targetName = data.input_name || getCleanLabel(data.name);
                            const input = this.inputs.find(i => i.name === targetName);
                            if (input) input.loraId = id;
                        }
                    }

                    resizeNodeKeepingWidth(this);
                };
        }
    }
});
