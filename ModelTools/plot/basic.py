from typing import Union

import pandas as pd
import altair as alt


class BasicPlot:
    def __init__(
        self,
        data       : Union[pd.DataFrame,alt.Chart],
        x          : str  = None,
        y          : str  = None,
        title      : str  = '',
        color      : str  = 'black',
        figure_size: list = [600,400]
    ) -> None:
        self.data = data 
        self.base = convert_to_chart(data=data) 
        self.x = x
        self.y = y 
        self.title = title
        self.color = color 
        self.figure_size = figure_size
    
    def set_attr(self,name,value):
        setattr(self,name,value)
        return self
    
    def __set_figure(func):
        def set_fig(self,*args,**kwargs):
            plot = func(self,*args,**kwargs)
            adj_plot = (
                plot 
                .properties(
                    title = self.title,
                    width = self.figure_size[0],
                    height = self.figure_size[1]
                )
            )
            return adj_plot 
        return set_fig
    
    @__set_figure
    def scatter(self):
        self.__check_param(type='xy')
        
        plot = (
            self.base 
            .mark_circle(
                color = self.color
            )
            .encode(
                x = alt.X(
                    self.x,
                    scale=alt.Scale(zero=False),
                    type='quantitative'
                ),
                y = alt.Y(
                    self.y,
                    scale=alt.Scale(zero=False),
                    type='quantitative'
                )
            )
        )
        return plot

    @__set_figure
    def hist(
        self,
        x='x',
        rotate=False,
        x_title=None,
        y_title=None,
    ):
        self.__check_param(x)
        x = self.__chose_param(x)
        
        X = alt.X(x,bin=True,type='quantitative',title=x_title)
        Y = alt.Y('count()',type='quantitative',title=y_title)
        if rotate == True:
            X,Y = Y,X
        plot = (
            self.base 
            .mark_bar(
                color = self.color
            )
            .encode(x = X,y = Y)
            .transform_bin('abc',field=x)
        )
        return plot
    
    @__set_figure
    def density(self,x='x',rotate=False):
        self.__check_param('x')
        x = self.__chose_param(x)
        
        X = alt.X(x,type='quantitative')
        Y = alt.Y('density',type='quantitative')
        if rotate is True:
            X,Y = Y,X
             
        plot = (
            self.base
            .transform_density(
                density=x,
                as_=[x,'density']
            )
            .encode(x = X,y = Y, order=x)
            .mark_line()
        )
        return plot
    
    @__set_figure
    def line(self):
        self.__check_param('xy')
        plot = (
            self.base 
            .encode(
                x = alt.X(self.x),
                y = alt.Y(self.y)
            )
            .mark_line()
        )
        return plot
    
    @__set_figure
    def abline(self,slope=None,intercept=None,name=None,name_position='right'):
        # 水平线
        if slope == 0 and intercept is not None:
            plot = (
                self.base
                .mark_rule(strokeDash=[12, 6],size=1,color=self.color)
                .encode(y=alt.datum(intercept))
            )
            if name is not None:
                self.__check_param('x')
                if name_position == 'right':
                    x = self.data.loc[:,self.x].max()
                elif name_position == 'left':
                    x = self.data.loc[:,self.x].min()
                text = (
                    plot
                    .mark_text(baseline='line-top',color=self.color)
                    .encode(text=alt.datum(name),x=alt.datum(x))
                )
                plot += text
        
        # 垂直性
        elif slope is None and intercept is not None:
            plot = (
                self.base
                .mark_rule(strokeDash=[12, 6],size=1,color=self.color)
                .encode(x=alt.datum(intercept))
            )
        
        # 斜线 
        elif slope != 0 and intercept is not None:
            self.__check_param('xy')
            x = self.data.loc[:,self.x]
            plot = (
                self.base
                .transform_calculate(as_='cal_y',calculate=f'datum.{self.x}*{slope}+{intercept}')
                .mark_line(color=self.color)
                .encode(
                    x = alt.X(self.x),
                    y = alt.Y('cal_y',type='quantitative')
                )
            )
        return plot
    
    @__set_figure
    def smooth(self,method='linear'):
        self.__check_param('xy')
        plot = (
            self.base
            .transform_regression(
                self.x,
                self.y,
                method=method
            )
            .encode(
                x = alt.X(self.x,type='quantitative'),
                y = alt.Y(self.y,type='quantitative')
            )
            .mark_line()
        )
        return plot

    def __check_param(self,type):
        if type == 'xy':
            if (self.x is None) or (self.y is None):
                raise Exception('Missing x or y')
        elif type == 'x':
            if self.x is None:
                raise Exception('Missing x')
    
    def __chose_param(self,x):
        if x == 'x':
            x = self.x 
        elif x == 'y':
            x = self.y 
        return x


def convert_to_chart(
    data:Union[pd.DataFrame,alt.Chart],
    chart_kwargs={}
) -> alt.Chart:
    if isinstance(data,pd.DataFrame):
        base = alt.Chart(data,**chart_kwargs)
    elif isinstance(data,alt.Chart):
        base = data 
        if len(chart_kwargs) > 0:
            raise Exception('WRONG')
    else:
        raise Exception('WRONG TYPE')
    return base