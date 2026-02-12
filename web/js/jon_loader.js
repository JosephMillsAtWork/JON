import { app } from "/scripts/app.js";

let CACHED_LORA_NAMES = [];

app.registerExtension({
    name: "JosephOddNodes.JonLoaders",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const targetNodes = ["JonLoader", "JonModelOnlyLoader"];

        if (targetNodes.includes(nodeData.name)) {

            // Fetch LoRA Names
            const resp = await fetch("/object_info/LoraLoader");
            const data = await resp.json();
            if (data && data.LoraLoader && data.LoraLoader.input.required.lora_name)
                CACHED_LORA_NAMES = data.LoraLoader.input.required.lora_name[0];

            const getCleanLabel = (path) => {
                if (!path || path === "Select LoRA...")
                    return "Strength";
                let parts = path.split("/");
                return parts[parts.length - 1].replace(/\.[^/.]+$/, "") + "_strength";
            };

            const updateHiddenConfig = (node) => {
                const state = {};
                if (node.widgets) {
                    node.widgets.forEach(w => {
                        if (w.name && w.name.startsWith("lora_name_")) {
                            const id = w.name.split("lora_name_")[1];
                            if (!state[id])
                                state[id] = {};
                            state[id].name = w.value;
                            if (node.inputs) {
                                const input = node.inputs.find(i => i.loraId === id);
                                if (input) state[id].input_name = input.name;
                            }
                        }
                        if (w.name && w.name.startsWith("lora_enabled_")) {
                            const id = w.name.split("lora_enabled_")[1];
                            if (!state[id])
                                state[id] = {};
                            state[id].enabled = w.value;
                        }
                    });
                }
                const configW = node.widgets.find(w => w.name === "lora_stack_json");
                if (configW)
                    configW.value = JSON.stringify(state);
            };

            const resizeNodeKeepingWidth = (node) => {
                const currentWidth = node.size[0];
                const minSize = node.computeSize();
                node.setSize([Math.max(currentWidth, minSize[0]), minSize[1]]);
            };

            // UI Element Creators
            const createSeparator = () => ({
                type: "separator", name: "separator",
                draw: (ctx, node, w, y) => {
                    ctx.strokeStyle = "#444"; ctx.beginPath(); ctx.moveTo(0, y+5); ctx.lineTo(w, y+5); ctx.stroke();
                },
                computeSize: (w) => [w, 10]
            });

            const createLabel = (text) => ({
                type: "label", name: "label_" + text,
                draw: (ctx, node, w, y) => {
                    ctx.fillStyle = "#aaa"; ctx.font = "bold 12px Arial"; ctx.fillText(text, 10, y + 15);
                },
                computeSize: (w) => [w, 20]
            });


            const toggleWidget = (node, name, enable) => {
                const w = node.widgets.find(w => w.name === name);
                if (w)
                    w.disabled = !enable;
            };

            const getVal = (node, name) => {
                const w = node.widgets.find(w => w.name === name);
                return w ? w.value : false;
            };

            const refreshVisibility = (node) => {
                // Model
                const modelType = getVal(node, "model_type");
                toggleWidget(node, "ckpt_name", false);
                toggleWidget(node, "unet_name", false);
                toggleWidget(node, "gguf_unet_name", false);

                if(modelType === "checkpoint")
                    toggleWidget(node, "ckpt_name", true);
                else if(modelType === "diffusion")
                    toggleWidget(node, "unet_name", true);
                else
                    toggleWidget(node, "gguf_unet_name", true);



                // Primary CLIP
                const ckNodeVal =  getVal(node, "clip_model_type");
                if(ckNodeVal === "gguf"){
                    toggleWidget(node, "clip_name", false);
                    toggleWidget(node, "gguf_clip_name", true);
                } else if(ckNodeVal === "safetensor"){
                    toggleWidget(node, "clip_name", true);
                    toggleWidget(node, "gguf_clip_name", false);
                }else{
                    // This must be a checkpoiunt
                    if(modelType !== "checkpoint")
                        console.warn("[JosephOddNodes.JonLoaders] " + modelType + " is not selected but but missing a file name")
                    toggleWidget(node, "clip_name", false);
                    toggleWidget(node, "gguf_clip_name", false);
                }

                // Dual Clip
                const daulChk =  getVal(node, "dual_clip");
                if(daulChk){
                    toggleWidget(node, "secondary_clip_model_type", true);
                    toggleWidget(node, "secondary_clip_type", true);

                    const other_ckNodeVal =  getVal(node, "secondary_clip_model_type");
                    if(other_ckNodeVal == "gguf"){
                        toggleWidget(node, "clip_name_2", false);
                        toggleWidget(node, "gguf_clip_name_2", true);
                    }else{
                        toggleWidget(node, "clip_name_2", true);
                        toggleWidget(node, "gguf_clip_name_2", false);

                    }
                }else{
                    toggleWidget(node, "secondary_clip_model_type", false);
                    toggleWidget(node, "secondary_clip_type", false);
                    toggleWidget(node, "clip_name_2", false);
                    toggleWidget(node, "gguf_clip_name_2", false);
                }

                // Secondary audio VAE
                const ckVaeVal =  getVal(node, "dual_vae");
                if(ckVaeVal){
                    toggleWidget(node, "vae_audio_name", true);
                    toggleWidget(node, "vae_audio_device", true);
                    toggleWidget(node, "vae_audio_dtype", true);
                }else{
                    toggleWidget(node, "vae_audio_name", false);
                    toggleWidget(node, "vae_audio_device", false);
                    toggleWidget(node, "vae_audio_dtype", false);
                }
            };

            // LoRA Slot
            const addLoRASlot = function(node, index, savedName, savedEnabled) {
                const id = index;
                const widgetName = `lora_name_${id}`;
                const initialLabel = getCleanLabel(savedName || CACHED_LORA_NAMES[0]);

                let existingInput = node.inputs ? node.inputs.find(i => i.name === initialLabel && !i.loraId) : null;
                if (existingInput) {
                    existingInput.loraId = id;
                } else {
                    node.addInput(initialLabel, "FLOAT");
                    node.inputs[node.inputs.length - 1].loraId = id;
                }

                const sepTop = createSeparator();
                node.widgets.push(sepTop);

                const isEnabled = (savedEnabled !== undefined) ? savedEnabled : true;
                const toggle = node.addWidget("toggle", "Enable", isEnabled, (value) => {
                    combo.disabled = !value;
                    btn.disabled = false; toggle.disabled = false;
                    updateHiddenConfig(node);
                });
                toggle.name = `lora_enabled_${id}`;

                const combo = node.addWidget("combo", widgetName, savedName || CACHED_LORA_NAMES[0], (value) => {
                    const newLabel = getCleanLabel(value);
                    const targetInput = node.inputs.find(i => i.loraId === id);
                    if (targetInput)
                        targetInput.name = newLabel;
                    updateHiddenConfig(node);
                }, { values: CACHED_LORA_NAMES });
                combo.disabled = !isEnabled;
                combo.label = "lora_name";

                const btn = node.addWidget("button", `Remove LoRA`, null, () => {
                    const inputIdx = node.inputs.findIndex(i => i.loraId === id);
                    if (inputIdx > -1)
                        node.removeInput(inputIdx);

                    [sepTop, toggle, combo, btn, sepBottom].forEach(w => {
                            const idx = node.widgets.indexOf(w);
                            if (idx > -1)
                                node.widgets.splice(idx, 1);
                        });

                        updateHiddenConfig(node);
                        resizeNodeKeepingWidth(node);
                        node.setDirtyCanvas(true, true);
                    });

                    const sepBottom = createSeparator();
                    node.widgets.push(sepBottom);

                    return { sepTop, toggle, combo, btn, sepBottom };
                };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function () {
                    const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                    // hide JSON
                    const configWidget = this.widgets.find(w => w.name === "lora_stack_json");
                    if (configWidget) {
                        configWidget.type = "hidden";
                        configWidget.computeSize = () => [0,0];
                    }
                    const jsonInputIdx = this.findInputSlot("lora_stack_json");
                    if (jsonInputIdx > -1)
                        this.removeInput(jsonInputIdx);

                    // JonLoader ?
                    if (nodeData.name === "JonLoader") {
                        const modeOptsIdx= this.widgets.findIndex(w => w.name === "model_type");
                            this.widgets.splice(modeOptsIdx, 0, createLabel("Loader Configuration"), createSeparator());

                        const fileStartIdx = this.widgets.findIndex(w => w.name === "ckpt_name");
                        if (fileStartIdx > -1)
                            this.widgets.splice(fileStartIdx, 0, createLabel("Model"), createSeparator());

                        const textEncoderIdx = this.widgets.findIndex(w => w.name === "clip_type");
                            this.widgets.splice(textEncoderIdx, 0, createLabel("Primary Text Encoder"), createSeparator());

                        const sec_textEncoderIdx = this.widgets.findIndex(w => w.name === "secondary_clip_model_type");
                            this.widgets.splice(sec_textEncoderIdx, 0, createLabel("Secondary Text Encoder"), createSeparator());

                        const vaeIdx = this.widgets.findIndex(w => w.name === "vae_name");
                            this.widgets.splice(vaeIdx, 0, createLabel("Primary VAE"), createSeparator());

                        const sec_vaeIdx = this.widgets.findIndex(w => w.name === "vae_audio_name");
                            this.widgets.splice(sec_vaeIdx, 0, createLabel("Secondary VAE"), createSeparator());

                    }

                    // JonModelOnlyLoader ?
                    if (nodeData.name === "JonModelOnlyLoader") {
                        this.widgets.splice(0, 0, createLabel("Model Configuration"), createSeparator());
                        const fileStartIdx = this.widgets.findIndex(w => w.name === "ckpt_name" || w.name === "gguf_unet_name" || w.name === "unet_name");
                        if (fileStartIdx > -1)
                            this.widgets.splice(fileStartIdx, 0, createLabel("File Selection"), createSeparator());
                    }

                    // Sage Attention
                    const sageIdx = this.widgets.findIndex(w => w.name === "sage_kernel");
                    if (sageIdx > -1)
                        this.widgets.splice(sageIdx, 0, createLabel("Sage Attention"), createSeparator());

                    // LoRA Stack
                    this.widgets.push(createLabel("LoRA Stack"));
                    this.widgets.push(createSeparator());
                    this.addWidget("button", "Add LoRA", null, () => {
                        const uniqueID = Date.now().toString().slice(-6);
                        addLoRASlot(this, uniqueID, null, true);
                        resizeNodeKeepingWidth(this);
                        updateHiddenConfig(this);
                    });

                    // Listeners
                    const triggers = ["model_type", "clip_model_type", "dual_clip", "dual_vae", "secondary_clip_model_type"];
                    triggers.forEach(t => {
                        const w = this.widgets.find(w => w.name === t);
                        if (w)
                            w.callback = () => { refreshVisibility(this); };
                    });

                    setTimeout(() => { refreshVisibility(this); }, 50);

                    return r;
                };

                //  onConfigure
                const onConfigure = nodeType.prototype.onConfigure;
                nodeType.prototype.onConfigure = function(w) {
                    if (onConfigure)
                        onConfigure.apply(this, arguments);

                    refreshVisibility(this);

                    let state = {};
                    if (w.widgets_values) {
                        for (const val of w.widgets_values) {
                            if (typeof val === 'string' && val.startsWith('{')) {
                                try {
                                    const parsed = JSON.parse(val);
                                    const firstKey = Object.keys(parsed)[0];
                                    if(firstKey && (parsed[firstKey].name || parsed[firstKey].enabled !== undefined)) {
                                        state = parsed;
                                        break;
                                    }
                                } catch(e) {}
                            }
                        }
                    }
                    for (const [id, data] of Object.entries(state)) {
                        addLoRASlot(this, id, data.name, data.enabled);
                    }
                    if (this.inputs) {
                        for (const [id, data] of Object.entries(state)) {
                            const targetName = data.input_name || getCleanLabel(data.name);
                            const input = this.inputs.find(i => i.name === targetName);
                            if (input)
                                input.loraId = id;
                        }
                    }
                    resizeNodeKeepingWidth(this);
                };
        }
    }
});
