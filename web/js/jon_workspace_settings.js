import { app } from "/scripts/app.js";

let CACHED_LORA_NAMES = [];

app.registerExtension({
    name: "JosephOddNodes.JonWorkflowSettings",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const targetNodes = ["JonWorkflowSettings"];

        if (targetNodes.includes(nodeData.name)) {
/*
            const resizeNodeKeepingWidth = (node) => {
                const currentWidth = node.size[0];
                const minSize = node.computeSize();
                node.setSize([Math.max(currentWidth, minSize[0]), minSize[1]]);
            };
*/
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
                // model
                const useImage1 = getVal(node, "use_first_img");
                toggleWidget(node, "img_1_name", useImage1);


                const useImage2 = getVal(node, "use_mid_img");
                toggleWidget(node, "img_2_name", useImage2);


                const useImage3 = getVal(node, "use_last_img");
                toggleWidget(node, "img_3_name", useImage3);
            };


            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                if (nodeData.name === "JonWorkflowSettings") {

                    this.widgets.splice(0, 0, createLabel("Enabled Images"), createSeparator());
                    const seedStartIdx = this.widgets.findIndex(w => w.name === "seed");
                    if (seedStartIdx > -1)
                        this.widgets.splice(seedStartIdx, 0, createLabel("Global Seed"), createSeparator());

                    const geoTimeStartIdx = this.widgets.findIndex(w => w.name === "aspect_ratio");
                    if (geoTimeStartIdx > -1)
                        this.widgets.splice(geoTimeStartIdx, 0, createLabel("Geometry & Time"), createSeparator());

                    const pos_promptStartIdx = this.widgets.findIndex(w => w.name === "positive_prompt");
                    if (pos_promptStartIdx > -1)
                        this.widgets.splice(pos_promptStartIdx, 0, createLabel("Positive Prompts"), createSeparator());

                    const neg_promptStartIdx = this.widgets.findIndex(w => w.name === "negative_prompt");
                    if (neg_promptStartIdx > -1)
                        this.widgets.splice(neg_promptStartIdx, 0, createLabel("Negative Prompts"), createSeparator());

                    const processStartIdx = this.widgets.findIndex(w => w.name === "resampling");
                    if (processStartIdx > -1)
                        this.widgets.splice(processStartIdx, 0, createLabel("Image Crop Settings"), createSeparator());


                    const vsecStartIdx = this.widgets.findIndex(w => w.name === "video_seconds");
                    if (vsecStartIdx > -1)
                        this.widgets.splice(vsecStartIdx, 0, createLabel("Video Length(Seconds)"), createSeparator());

                    const upsccaleStartIdx = this.widgets.findIndex(w => w.name === "upscale_enabled");
                    if (upsccaleStartIdx > -1)
                        this.widgets.splice(upsccaleStartIdx, 0, createLabel("Enable Latent Upscaler (If Supported)"), createSeparator());
                }


                const triggers = ["use_first_img", "use_mid_img", "use_last_img"];
                triggers.forEach(t => {
                    const w = this.widgets.find(w => w.name === t);
                    if (w)
                        w.callback = () => { refreshVisibility(this); };
                });
                setTimeout(() => { refreshVisibility(this); }, 50);

            };


            //  onConfigure
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(w) {
                if (onConfigure)
                    onConfigure.apply(this, arguments);

                refreshVisibility(this);
                // resizeNodeKeepingWidth(this);
            };

        }
    }
});
